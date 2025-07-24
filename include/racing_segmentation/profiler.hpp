#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "rclcpp/rclcpp.hpp"

class SimpleProfiler {
public:
    // 开始计时
    void start(const std::string& name) {
        start_times_[name] = std::chrono::steady_clock::now();
    }

    // 停止计时并累加耗时
    void stop(const std::string& name) {
        auto end_time = std::chrono::steady_clock::now();
        auto it = start_times_.find(name);
        if (it != start_times_.end()) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - it->second);
            accumulated_times_[name] += duration.count();
        }
    }

    // 增加计数器
    void count(const std::string& name, int count = 1) {
        counters_[name] += count;
    }

    // 重置所有数据
    void reset() {
        accumulated_times_.clear();
        counters_.clear();
    }
    
    // 打印报告
    void report(const rclcpp::Logger& logger) {
        std::stringstream ss;
        ss << "\n--- Performance Report (avg over " << get_main_counter() << " frames) ---\n";
        
        for (const auto& pair : accumulated_times_) {
            const std::string& name = pair.first;
            long long total_micros = pair.second;
            
            auto counter_it = counters_.find(name);
            int num_calls = (counter_it != counters_.end() && counter_it->second > 0) ? counter_it->second : 1;

            double avg_ms = static_cast<double>(total_micros) / num_calls / 1000.0;
            
            ss << std::fixed << std::setprecision(2)
               << "  " << std::left << std::setw(30) << name << ": " 
               << std::right << std::setw(7) << avg_ms << " ms\n";
        }
        ss << "--------------------------------------------------";
        
        RCLCPP_INFO(logger, "%s", ss.str().c_str());
    }

private:
    std::map<std::string, std::chrono::steady_clock::time_point> start_times_;
    std::map<std::string, long long> accumulated_times_;
    std::map<std::string, int> counters_;

    int get_main_counter(){
        auto it = counters_.find("Total_Callback");
        if(it != counters_.end()){
            return it->second;
        }
        return 0;
    }
};

#endif // PROFILER_H
