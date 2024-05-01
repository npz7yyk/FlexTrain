#pragma once
#include <cassert>
#include <thread>
#include <map>
#include <utility>

typedef uint64_t thread_handle_t;

class ThreadHelper {
public:
    thread_handle_t getNextHandle() {
        ++event_id_;
        if (event_id_ == 0)
            ++event_id_;
        return event_id_;
    }

    template<typename F, typename ...Args>
    void start(thread_handle_t event_id, const F &func, Args &&...args) {
        assert(events_.count(event_id) == 0);
        events_[event_id] = std::thread{std::forward<F>(func), std::forward<Args>(args)...};
    }

    template<class Class, typename F, typename... Args>
    thread_handle_t start(F *func, Class *cls, Args &&...args) {
        ++event_id_;
        if (event_id_ == 0)
            ++event_id_;
        events_[event_id_] = std::thread(cls->*func, std::forward<Args>(args)...);
        return event_id_;
    }


    void sync(const thread_handle_t event_id) {
        assert(events_.count(event_id));
        events_.at(event_id).join();
        events_.erase(event_id);
    }

    static ThreadHelper &GetInstance() {
        static ThreadHelper thread_helper{};
        return thread_helper;
    }

private:
    thread_handle_t event_id_{0};
    std::map<thread_handle_t, std::thread> events_;
};