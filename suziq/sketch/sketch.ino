// LED matrix feedback for the Oracle: breathing while idle, a quick flash
// while a photo is being processed, then whatever static frame or scroll
// Python pushes next (dice values, X's for a miss, the poem scrolling) held
// indefinitely until Python calls resume_breathing.
//
// Bridge providers run on a separate thread from loop() - state shared
// between them and loop() is protected by a mutex.
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>
#include <zephyr/kernel.h>
#include <math.h>

Arduino_LED_Matrix matrix;

const uint8_t FRAME_SIZE = 8 * 13;

K_MUTEX_DEFINE(state_mtx);

// RESULT holds indefinitely now - Python owns the whole display lifecycle
// (numbers hang on screen through the LLM/TTS "thinking" time, then the
// poem scrolls, then Python explicitly calls resume_breathing when done or
// cancelled) rather than the board guessing a fixed hold duration.
enum DisplayState { BREATHING, BUSY, RESULT };
static DisplayState state = BREATHING;
static uint8_t resultFrame[FRAME_SIZE] = {0};

static unsigned long lastBreathUpdate = 0;
static float breathPhase = 0;

void setup() {
    matrix.begin();
    matrix.setGrayscaleBits(3);  // 0-7 brightness levels
    matrix.clear();

    Bridge.begin();
    Bridge.provide("show_busy", show_busy);
    Bridge.provide("show_result", show_result);
    Bridge.provide("resume_breathing", resume_breathing);
}

void loop() {
    k_mutex_lock(&state_mtx, K_FOREVER);
    DisplayState s = state;
    k_mutex_unlock(&state_mtx);

    if (s == RESULT) {
        k_mutex_lock(&state_mtx, K_FOREVER);
        matrix.draw(resultFrame);
        k_mutex_unlock(&state_mtx);
    } else if (s == BUSY) {
        // Alternating brightness - a plain "still working" flash.
        uint8_t level = ((millis() / 150) % 2 == 0) ? 7 : 1;
        uint8_t frame[FRAME_SIZE];
        for (int i = 0; i < FRAME_SIZE; i++) frame[i] = level;
        k_mutex_lock(&state_mtx, K_FOREVER);
        matrix.draw(frame);
        k_mutex_unlock(&state_mtx);
    } else {  // BREATHING
        unsigned long now = millis();
        if (now - lastBreathUpdate > 30) {
            lastBreathUpdate = now;
            breathPhase += 0.05;
            uint8_t level = (uint8_t)(((sin(breathPhase) + 1.0) / 2.0) * 7);
            uint8_t frame[FRAME_SIZE];
            for (int i = 0; i < FRAME_SIZE; i++) frame[i] = level;
            k_mutex_lock(&state_mtx, K_FOREVER);
            matrix.draw(frame);
            k_mutex_unlock(&state_mtx);
        }
    }
    delay(5);
}

// --- Bridge providers --------------------------------------------------------

void show_busy() {
    k_mutex_lock(&state_mtx, K_FOREVER);
    state = BUSY;
    k_mutex_unlock(&state_mtx);
}

void show_result(std::vector<uint8_t> frame) {
    if (frame.empty()) return;
    k_mutex_lock(&state_mtx, K_FOREVER);
    size_t len = min(frame.size(), (size_t)FRAME_SIZE);
    memcpy(resultFrame, frame.data(), len);
    state = RESULT;
    k_mutex_unlock(&state_mtx);
}

// Lets the Python side end a result display - once a scroll finishes, or a
// miss's static X's have shown long enough.
void resume_breathing() {
    k_mutex_lock(&state_mtx, K_FOREVER);
    state = BREATHING;
    k_mutex_unlock(&state_mtx);
}
