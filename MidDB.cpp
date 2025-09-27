#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include <condition_variable>
#include <thread>
#include <filesystem>
#include "httplib.h"
#include "json.hpp"
#include "hnswlib/hnswlib.h"

using namespace std;
using json = nlohmann::json;
namespace fs = std::filesystem;

// --- Data Structures ---
struct Record {
    unordered_map<string,string> fields;
    vector<float> embedding;
    size_t label;
};

struct Table {
    unordered_map<string,Record> records;
    unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    unordered_map<size_t,string> labelToID;
    size_t nextLabel = 0;
    int dim = 0;

    // Structured field index: fieldName -> fieldValue -> set(recordIDs)
    unordered_map<string, unordered_map<string, unordered_set<string>>> fieldIndex;
};

// --- MidDB Class ---
class MidDB {
private:
    unordered_map<string,Table> tables;
    string storageDir = "data";
    mutable shared_mutex dbMutex; // for shared read access

    // Async insert
    struct InsertTask { string tableName, recordID; unordered_map<string,string> fields; vector<float> embedding; };
    queue<InsertTask> insertQueue;
    mutex queueMutex;               // only for queue + condition_variable
    condition_variable cv;
    bool stopWorker = false;
    thread workerThread;

    string tableFile(const string &tableName) { return storageDir + "/" + tableName + ".json"; }
    string indexFile(const string &tableName) { return storageDir + "/" + tableName + ".index"; }

    void worker() {
        vector<InsertTask> batch;
        while (true) {
            {
                unique_lock<mutex> lock(queueMutex);
                cv.wait_for(lock, chrono::seconds(5), [&]{ return !insertQueue.empty() || stopWorker; });
                if (stopWorker && insertQueue.empty()) break;
                batch.clear();
                while (!insertQueue.empty() && batch.size() < 100) {
                    batch.push_back(insertQueue.front());
                    insertQueue.pop();
                }
            }
            for (auto &task : batch) processInsert(task);
            saveAllTables();
        }
    }

    void processInsert(const InsertTask &task) {
        unique_lock<shared_mutex> lock(dbMutex);

        if (tables.find(task.tableName) == tables.end())
            createTable(task.tableName, task.embedding.size());

        auto &table = tables[task.tableName];
        if (!table.index) {
            auto space = new hnswlib::L2Space(task.embedding.size());
            table.index.reset(new hnswlib::HierarchicalNSW<float>(space, 20000));
        }

        size_t label;
        auto recIt = table.records.find(task.recordID);
        if (recIt != table.records.end()) {
            // Update existing record (preserve label)
            label = recIt->second.label;
            recIt->second.fields = task.fields;
            recIt->second.embedding = task.embedding;
        } else {
            // Insert new record
            label = table.nextLabel++;
            table.records[task.recordID] = {task.fields, task.embedding, label};
        }
        table.labelToID[label] = task.recordID;

        // Update structured index
        for (auto &[key,val] : task.fields)
            table.fieldIndex[key][val].insert(task.recordID);

        // Add to HNSW index
        table.index->addPoint(task.embedding.data(), label);

        cout << "[INFO] Inserted/Updated " << task.recordID << " into " << task.tableName << " (label=" << label << ")\n";
    }

    void saveAllTables() {
        shared_lock<shared_mutex> lock(dbMutex);
        for (auto &p : tables) {
            saveTable(p.first);
            saveIndex(p.first);
        }
    }

public:
    MidDB() {
        fs::create_directories(storageDir);
        for (auto &p : fs::directory_iterator(storageDir))
            if (p.path().extension() == ".json")
                loadTable(p.path().stem().string());
        workerThread = thread([this]{ worker(); });
    }

    ~MidDB() {
        {
            lock_guard<mutex> lock(queueMutex);
            stopWorker = true;
        }
        cv.notify_all();
        if(workerThread.joinable()) workerThread.join();
    }

    void createTable(const string &tableName, int dim = 0) {
        if (tables.find(tableName) != tables.end()) return;
        Table t; t.dim = dim;
        tables[tableName] = std::move(t);
    }

    void insert(const string &tableName, const string &recordID,
                const unordered_map<string,string> &fields,
                const vector<float> &embedding) {
        {
            lock_guard<mutex> lock(queueMutex);
            insertQueue.push({tableName, recordID, fields, embedding});
        }
        cv.notify_one();
    }

    void update(const string &tableName, const string &recordID,
                const unordered_map<string,string> &fields,
                const vector<float> &embedding) {
        insert(tableName, recordID, fields, embedding); // upsert via insert
    }

    void remove(const string &tableName, const string &recordID) {
        unique_lock<shared_mutex> lock(dbMutex);
        if (tables.find(tableName) == tables.end()) return;
        auto &table = tables[tableName];
        auto it = table.records.find(recordID);
        if (it == table.records.end()) return;

        size_t label = it->second.label;
        // Remove from main records
        table.records.erase(it);
        table.labelToID.erase(label);

        // Remove from structured index
        for (auto &[key,val] : it->second.fields) {
            auto fIt = table.fieldIndex.find(key);
            if(fIt != table.fieldIndex.end()) {
                auto vIt = fIt->second.find(val);
                if(vIt != fIt->second.end()) vIt->second.erase(recordID);
            }
        }

        // Soft delete from HNSW (ghost label will exist)
        if(table.index) table.index->markDelete(label);

        cout << "[INFO] Deleted " << recordID << " from " << tableName << "\n";
    }

    vector<string> queryField(const string &tableName, const string &field, const string &value) const {
        vector<string> result;
        shared_lock<shared_mutex> lock(dbMutex);
        if (tables.find(tableName) == tables.end()) return result;
        const auto &table = tables.at(tableName);
        auto fit = table.fieldIndex.find(field);
        if (fit != table.fieldIndex.end()) {
            auto vit = fit->second.find(value);
            if (vit != fit->second.end()) {
                result.reserve(vit->second.size());
                for (const auto &id : vit->second) result.push_back(id);
                sort(result.begin(), result.end());
            }
        }
        return result;
    }

    vector<string> queryEmbedding(const string &tableName, const vector<float> &embedding, int topK=3) const {
        vector<string> result;
        shared_lock<shared_mutex> lock(dbMutex);
        if (tables.find(tableName) == tables.end()) return result;
        const auto &table = tables.at(tableName);
        if (!table.index) return result;

        auto labels = table.index->searchKnn(embedding.data(), topK);
        while (!labels.empty()) {
            auto item = labels.top(); labels.pop();
            auto it = table.labelToID.find(item.second);
            if (it != table.labelToID.end()) result.push_back(it->second);
        }
        return result;
    }

    vector<string> queryHybrid(const string &tableName,
                               const string &field, const string &value,
                               const vector<float> &embedding, int topK=3) const {
        auto filteredIDs = queryField(tableName, field, value);
        if (filteredIDs.empty()) return {};

        auto candidateIDs = queryEmbedding(tableName, embedding, topK*10);
        unordered_set<string> filterSet(filteredIDs.begin(), filteredIDs.end());

        vector<string> final;
        for (auto &id : candidateIDs)
            if (filterSet.count(id)) final.push_back(id);
        if (final.size() > (size_t)topK) final.resize(topK);
        return final;
    }

    void saveTable(const string &tableName) {
        auto &table = tables[tableName];
        json j;
        for (auto &[id, rec] : table.records) {
            j[id]["fields"] = rec.fields;
            j[id]["embedding"] = rec.embedding;
            j[id]["label"] = rec.label;
        }
        ofstream out(tableFile(tableName));
        out << j.dump(2);
    }

    void saveIndex(const string &tableName) {
        auto &table = tables[tableName];
        if (table.index) table.index->saveIndex(indexFile(tableName));
    }

    void loadTable(const string &tableName) {
        ifstream in(tableFile(tableName));
        if (!in.is_open()) return;

        json j; in >> j;
        Table t;
        for (auto &[id, rec] : j.items()) {
            Record r;
            r.fields = rec["fields"].get<unordered_map<string,string>>();
            r.embedding = rec["embedding"].get<vector<float>>();
            r.label = rec["label"].get<size_t>();
            t.records[id] = r;
            t.labelToID[r.label] = id;
            for (auto &[key,val] : r.fields)
                t.fieldIndex[key][val].insert(id);
            if (t.dim==0) t.dim = r.embedding.size();
            if (r.label >= t.nextLabel) t.nextLabel = r.label+1;
        }
        if (ifstream(indexFile(tableName)).good() && t.dim>0) {
            auto space = new hnswlib::L2Space(t.dim);
            t.index.reset(new hnswlib::HierarchicalNSW<float>(space, indexFile(tableName)));
        }
        tables[tableName] = std::move(t);
    }
};

// --- REST API ---
int main() {
    MidDB db;
    httplib::Server svr;

    // --- CRUD Endpoints ---
    svr.Post("/insert", [&db](const httplib::Request &req, httplib::Response &res){
        try {
            auto j = json::parse(req.body);
            db.insert(j["table"], j["id"],
                      j["fields"].get<unordered_map<string,string>>(),
                      j["embedding"].get<vector<float>>());
            res.set_content("{\"status\":\"ok\"}", "application/json");
        } catch(exception &e){
            res.status = 400;
            res.set_content("{\"error\":\""+string(e.what())+"\"}", "application/json");
        }
    });

    svr.Post("/update", [&db](const httplib::Request &req, httplib::Response &res){
        try {
            auto j = json::parse(req.body);
            db.update(j["table"], j["id"],
                      j["fields"].get<unordered_map<string,string>>(),
                      j["embedding"].get<vector<float>>());
            res.set_content("{\"status\":\"ok\"}", "application/json");
        } catch(exception &e){
            res.status = 400;
            res.set_content("{\"error\":\""+string(e.what())+"\"}", "application/json");
        }
    });

    svr.Post("/delete", [&db](const httplib::Request &req, httplib::Response &res){
        try {
            auto j = json::parse(req.body);
            db.remove(j["table"], j["id"]);
            res.set_content("{\"status\":\"ok\"}", "application/json");
        } catch(exception &e){
            res.status = 400;
            res.set_content("{\"error\":\""+string(e.what())+"\"}", "application/json");
        }
    });

    // --- Query Endpoints ---
    svr.Get(R"(/queryField/(\w+))", [&db](const httplib::Request &req, httplib::Response &res){
        string table = req.matches[1];
        string field = req.get_param_value("field");
        string value = req.get_param_value("value");
        auto ids = db.queryField(table,field,value);
        res.set_content(json(ids).dump(),"application/json");
    });

    svr.Post(R"(/queryEmbedding/(\w+))", [&db](const httplib::Request &req, httplib::Response &res){
        try {
            string table = req.matches[1];
            auto j = json::parse(req.body);
            vector<float> emb = j["embedding"].get<vector<float>>();
            int topK = j.value("topK",3);
            auto ids = db.queryEmbedding(table,emb,topK);
            res.set_content(json(ids).dump(),"application/json");
        } catch(exception &e){
            res.status = 400;
            res.set_content("{\"error\":\""+string(e.what())+"\"}", "application/json");
        }
    });

    svr.Post(R"(/queryHybrid/(\w+))", [&db](const httplib::Request &req, httplib::Response &res){
        try {
            string table = req.matches[1];
            auto j = json::parse(req.body);
            string field = j["field"];
            string value = j["value"];
            vector<float> emb = j["embedding"].get<vector<float>>();
            int topK = j.value("topK",3);
            auto ids = db.queryHybrid(table,field,value,emb,topK);
            res.set_content(json(ids).dump(),"application/json");
        } catch(exception &e){
            res.status = 400;
            res.set_content("{\"error\":\""+string(e.what())+"\"}", "application/json");
        }
    });

    cout << "MidDB (structured + semantic + hybrid) running at http://localhost:8080\n";
    svr.listen("0.0.0.0",8080);
}
