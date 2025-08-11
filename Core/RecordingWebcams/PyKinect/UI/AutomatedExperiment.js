document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const pid = urlParams.get('pid');
    if (!pid) {
        alert('No participant ID provided!');
        return;
    }
    const TOTAL_DURATION = 20; // seconds per gesture (5s reading + 15s performance)
    const READING_TIME = 5; // seconds for reading
    const PERFORMANCE_TIME = 15; // seconds for performing
    const BREAK_DURATION = 120; // 2 minutes break
    const GESTURES_PER_SET = 25; // gestures per set before break
    let gestureSequence = [];
    let currentGestureIndex = 0;
    let timerInterval = null;

    // Show start screen, hide experiment screen
    document.getElementById('start-screen').style.display = 'flex';
    document.getElementById('experiment-screen').style.display = 'none';
    document.getElementById('complete-screen').style.display = 'none';
    document.getElementById('break-screen').style.display = 'none';

    // Only start experiment when button is clicked
    document.getElementById('start-btn').addEventListener('click', () => {
        document.getElementById('start-screen').style.display = 'none';
        document.getElementById('experiment-screen').style.display = 'block';
        fetchAndStartExperiment();
    });

    function fetchAndStartExperiment() {
        fetch(`/schedules/participant_${pid}_schedule.json`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(schedule => {
                gestureSequence = schedule.gesture_sequence;
                startExperiment();
            })
            .catch(err => {
                console.error('Error loading schedule:', err);
                alert('Could not load gesture schedule for this participant. Please check the console for details.');
            });
    }

    function startExperiment() {
        showGesture(0);
    }

    function showGesture(index) {
        if (index >= gestureSequence.length) {
            document.getElementById('experiment-screen').style.display = 'none';
            document.getElementById('complete-screen').style.display = 'block';
            setTimeout(() => {
                document.getElementById('complete-screen').style.display = 'none';
                document.getElementById('thankyou-screen').style.display = 'block';
            }, 2000); // Show thank you screen after 2 seconds
            return;
        }

        // Check if we need a break (after every 20 gestures, but not at the very end)
        if (index > 0 && index % GESTURES_PER_SET === 0 && index < gestureSequence.length) {
            showBreak(index);
            return;
        }

        const gesture = gestureSequence[index];
        const currentSet = Math.floor(index / GESTURES_PER_SET) + 1;
        const totalSets = Math.ceil(gestureSequence.length / GESTURES_PER_SET);
        
        document.getElementById('gesture-display').textContent = gesture;
        document.getElementById('gesture-count').textContent = 
            `Gesture ${index + 1} of ${gestureSequence.length} (Set ${currentSet} of ${totalSets})`;
        
        // Start with reading phase (3 seconds)
        let timeLeft = TOTAL_DURATION;
        let phase = 'reading';
        updateDisplay(gesture, phase, timeLeft, 0);
        
        timerInterval = setInterval(() => {
            timeLeft--;
            
            // Determine current phase
            if (timeLeft > PERFORMANCE_TIME) {
                phase = 'reading';
            } else {
                phase = 'performing';
                // Log gesture start when performance phase begins
                if (timeLeft === PERFORMANCE_TIME) {
                    logGesture(gesture, index + 1);
                }
            }
            
            updateDisplay(gesture, phase, timeLeft);
            
            if (timeLeft <= 0) {
                clearInterval(timerInterval);
                showGesture(index + 1);
            }
        }, 1000);
    }

    function showBreak(nextGestureIndex) {
        document.getElementById('experiment-screen').style.display = 'none';
        document.getElementById('break-screen').style.display = 'block';
        
        const currentSet = Math.floor(nextGestureIndex / GESTURES_PER_SET);
        const totalSets = Math.ceil(gestureSequence.length / GESTURES_PER_SET);
        
        document.getElementById('break-title').textContent = `Break - Set ${currentSet} Complete`;
        document.getElementById('break-info').textContent = 
            `You have completed ${currentSet} of ${totalSets} sets. Take a 2-minute break.`;
        
        let breakTimeLeft = BREAK_DURATION;
        updateBreakTimer(breakTimeLeft);
        
        timerInterval = setInterval(() => {
            breakTimeLeft--;
            updateBreakTimer(breakTimeLeft);
            
            if (breakTimeLeft <= 0) {
                clearInterval(timerInterval);
                document.getElementById('break-screen').style.display = 'none';
                document.getElementById('experiment-screen').style.display = 'block';
                showGesture(nextGestureIndex);
            }
        }, 1000);
    }

    function updateBreakTimer(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        const timeString = `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
        document.getElementById('break-timer').textContent = timeString;
    }

    function updateDisplay(gesture, phase, timeLeft) {
        let timerText = '';
        let phaseText = '';
        let progress = 0;
        
        if (phase === 'reading') {
            const readingTimeLeft = timeLeft - PERFORMANCE_TIME;
            timerText = readingTimeLeft;
            phaseText = 'Reading Time';
            // Progress bar empties during reading (3,2,1)
            progress = 0;
        } else {
            const performanceTimeLeft = timeLeft;
            timerText = performanceTimeLeft;
            phaseText = 'Performance Time';
            // Progress bar fills during performance (1,2,3,4,5,6,7)
            progress = ((PERFORMANCE_TIME - performanceTimeLeft) / PERFORMANCE_TIME) * 100;
        }
        
        document.getElementById('timer').textContent = timerText;
        document.getElementById('phase-indicator').textContent = phaseText;
        document.getElementById('progress-fill').style.width = `${progress}%`;
    }

    function logGesture(gesture, gestureIndex) {
        fetch('/log_gesture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pid: pid,
                gesture: gesture,
                gesture_index: gestureIndex,
                timestamp: new Date().toISOString()
            })
        }).catch(err => {
            console.error('Error logging gesture:', err);
        });
    }
}); 