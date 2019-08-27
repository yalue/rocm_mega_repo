//
// Copyright (c) 2008,2010 Advanced Micro Devices, Inc. All rights reserved.
//

#include "thread/semaphore.hpp"
#include "thread/thread.hpp"

#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else  // !_WIN32
#include <semaphore.h>
#include <errno.h>
#endif  // !_WIN32

namespace amd {

Semaphore::Semaphore() : state_(0) {
#ifdef _WIN32
  handle_ = static_cast<void*>(CreateSemaphore(NULL, 0, LONG_MAX, NULL));
  assert(handle_ != NULL && "CreateSemaphore failed");
#else   // !_WIN32
  if (sem_init(&sem_, 0, 0) != 0) {
    fatal("sem_init() failed");
  }
#endif  // !_WIN32
}

Semaphore::~Semaphore() {
#ifdef _WIN32
  if (!CloseHandle(static_cast<HANDLE>(handle_))) {
    fatal("CloseHandle() failed");
  }
#else   // !_WIN32
  if (sem_destroy(&sem_) != 0) {
    fatal("sem_destroy() failed");
  }
#endif  // !WIN32
}

void Semaphore::post() {
  int state = state_.load(std::memory_order_relaxed);
  for (;;) {
    if (state > 0) {
      int newstate = state_.load(std::memory_order_acquire);
      if (state == newstate) {
        return;
      }
      state = newstate;
      continue;
    }
    if (state_.compare_exchange_weak(state, state + 1, std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
      break;
    }
  }

  if (state < 0) {
// We have threads waiting on this event.
#ifdef _WIN32
    ReleaseSemaphore(static_cast<HANDLE>(handle_), 1, NULL);
#else  // !_WIN32
    if (0 != sem_post(&sem_)) {
      fatal("sem_post() failed");
    }
#endif  // !_WIN32
  }
}

void Semaphore::wait() {
  if (state_-- > 0) {
    return;
  }

#ifdef _WIN32
  if (WAIT_OBJECT_0 != WaitForSingleObject(static_cast<HANDLE>(handle_), INFINITE)) {
    fatal("WaitForSingleObject failed");
  }
#else   // !_WIN32
  while (0 != sem_wait(&sem_)) {
    if (EINTR != errno) {
      fatal("sem_wait() failed");
    }
  }
#endif  // !_WIN32
}

}  // namespace amd
