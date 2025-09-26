#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "httplib.h"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

// --- Data Structures ---

struct Record {
    unordered_map<string, string> fields; // structured data
    vector<float> embedding; // semantic vector
};

struct Table {
    unordered_map<string, Record> records; // record_id -> Record
};

// MidDB class
class MidDB {
private:
    unordered_map<string, Table> tables; // table_name -> Table

public:
    // Add a table dynamically
    void createTable(const string &tableName) {
        if (tables.find(tableName) == tables.end()) {
            tables[tableName] = Table();
        }
    }

    // Insert a record
    void insert(const string &tableName, const string &recordID,
                const unordered_map<string, string> &fields,
                const vector<float> &embedding) {
        createTable(tableName);
        tables[tableName].records[recordID] = {fields, embedding};
    }

    // Structured query: find records by field value
    vector<string> queryField(const string &tableName,
                                        const string &field,
                                        const string &value) {
        vector<string> result;
        if (tables.find(tableName) == tables.end()) return result;
        for (auto &[id, rec] : tables[tableName].records) {
            if (rec.fields.find(field) != rec.fields.end() &&
                rec.fields[field] == value) {
                result.push_back(id);
            }
        }
        return result;
    }

    // Semantic query: nearest neighbor search
    vector<string> queryEmbedding(const string &tableName,
                                            const vector<float> &embedding,
                                            int topK=3) {
        struct Score {
            string id;
            float dist;
        };
        vector<Score> scores;
        if (tables.find(tableName) == tables.end()) return {};
        for (auto &[id, rec] : tables[tableName].records) {
            if (!rec.embedding.empty() && rec.embedding.size() == embedding.size()) {
                float sum = 0.0f;
                for (size_t i=0; i<embedding.size(); i++) {
                    float diff = rec.embedding[i] - embedding[i];
                    sum += diff * diff;
                }
                scores.push_back({id, sqrt(sum)});
            }
        }
        sort(scores.begin(), scores.end(), [](const Score &a, const Score &b){
            return a.dist < b.dist;
        });
        vector<string> result;
        for (int i=0; i<min(topK,(int)scores.size()); i++) {
            result.push_back(scores[i].id);
        }
        return result;
    }
};

// --- Main REST API ---

int main() {
    MidDB db;
    httplib::Server svr;

    // Insert record
    svr.Post("/insert", [&db](const httplib::Request& req, httplib::Response& res){
        auto j = json::parse(req.body);
        string table = j["table"];
        string id = j["id"];
        unordered_map<string, string> fields = j["fields"].get<unordered_map<string,string>>();
        vector<float> embedding = j["embedding"].get<vector<float>>();
        db.insert(table, id, fields, embedding);
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // Structured query
    svr.Get(R"(/queryField/(\w+))", [&db](const httplib::Request& req, httplib::Response& res){
        string table = req.matches[1];
        string field = req.get_param_value("field");
        string value = req.get_param_value("value");
        auto ids = db.queryField(table, field, value);
        res.set_content(json(ids).dump(), "application/json");
    });

    // Semantic query
    svr.Post(R"(/queryEmbedding/(\w+))", [&db](const httplib::Request& req, httplib::Response& res){
        string table = req.matches[1];
        auto j = json::parse(req.body);
        vector<float> embedding = j["embedding"].get<vector<float>>();
        int topK = j.value("topK", 3);
        auto ids = db.queryEmbedding(table, embedding, topK);
        res.set_content(json(ids).dump(), "application/json");
    });

    cout << "MidDB server running at http://localhost:8080\n";
    svr.listen("0.0.0.0", 8080);
}
