#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <mutex>
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
};

// --- MidDB class ---
class MidDB {
private:
    unordered_map<string,Table> tables;
    string storageDir = "data";
    mutable mutex dbMutex;

    // Async insert queue
    struct InsertTask { string tableName, recordID; unordered_map<string,string> fields; vector<float> embedding; };
    queue<InsertTask> insertQueue;
    condition_variable cv;
    bool stopWorker = false;
    thread workerThread;

    string tableFile(const string &tableName) { return storageDir + "/" + tableName + ".json"; }
    string indexFile(const string &tableName) { return storageDir + "/" + tableName + ".index"; }

    void worker() {
        while (true) {
            InsertTask task;
            {
                unique_lock<mutex> lock(dbMutex);
                cv.wait(lock, [this]{ return !insertQueue.empty() || stopWorker; });
                if (stopWorker && insertQueue.empty()) break;
                task = insertQueue.front(); insertQueue.pop();
            }
            processInsert(task);
        }
    }

    void processInsert(const InsertTask &task) {
        lock_guard<mutex> lock(dbMutex);
        if (tables.find(task.tableName) == tables.end())
            createTable(task.tableName, task.embedding.size());
        auto &table = tables[task.tableName];
        if (!table.index) {
            auto space = new hnswlib::L2Space(task.embedding.size());
            table.index.reset(new hnswlib::HierarchicalNSW<float>(space, 20000));
        }
        size_t label = table.nextLabel++;
        table.records[task.recordID] = {task.fields, task.embedding, label};
        table.labelToID[label] = task.recordID;
        table.index->addPoint(task.embedding.data(), label);

        saveTable(task.tableName);
        saveIndex(task.tableName);
        cout << "[INFO] Inserted " << task.recordID << " into " << task.tableName << endl;
    }

public:
    MidDB() {
        fs::create_directories(storageDir);

        // Load existing tables
        for (auto &p : fs::directory_iterator(storageDir))
            if (p.path().extension() == ".json")
                loadTable(p.path().stem().string());

        // Start worker thread
        workerThread = thread([this]{ worker(); });
    }

    ~MidDB() {
        {
            lock_guard<mutex> lock(dbMutex);
            stopWorker = true;
        }
        cv.notify_all();
        workerThread.join();
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
            lock_guard<mutex> lock(dbMutex);
            insertQueue.push({tableName, recordID, fields, embedding});
        }
        cv.notify_one();
    }

    vector<string> queryField(const string &tableName, const string &field, const string &value) const {
        vector<string> result;
        lock_guard<mutex> lock(dbMutex);
        if (tables.find(tableName) == tables.end()) return result;
        for (auto &[id, rec] : tables.at(tableName).records)
            if (rec.fields.find(field) != rec.fields.end() && rec.fields.at(field) == value)
                result.push_back(id);
        return result;
    }

    vector<string> queryEmbedding(const string &tableName, const vector<float> &embedding, int topK=3) const {
        vector<string> result;
        lock_guard<mutex> lock(dbMutex);
        if (tables.find(tableName) == tables.end()) return result;
        auto &table = tables.at(tableName);
        if (!table.index) return result;
        auto labels = table.index->searchKnn(embedding.data(), topK);
        while (!labels.empty()) {
            auto item = labels.top(); labels.pop();
            auto it = table.labelToID.find(item.second);
            if (it != table.labelToID.end()) result.push_back(it->second);
        }
        return result;
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
            if (t.dim == 0) t.dim = r.embedding.size();
            if (r.label >= t.nextLabel) t.nextLabel = r.label+1;
        }
        if (ifstream(indexFile(tableName)).good() && t.dim > 0) {
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

    cout << "MidDB (production-ready, async inserts, persistent HNSW) running at http://localhost:8080\n";
    svr.listen("0.0.0.0",8080);
}
