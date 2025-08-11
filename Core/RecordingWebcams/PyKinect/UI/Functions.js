
let data = "Timestamp;Label \n";
let dataHTML = "Timestamp;Label <br/>";

currentElement = null;

/**
 * Add a new Label to the log.  
 * 
 * @param {any} element The Button that was clicked
 */
function newLabel(element) {

    // There is no current label
    if (currentElement == null){    
        element.className = "LabelKlicked";
        currentElement = element;
        console.log("New Label: " + element.textContent);
        addLine(element.textContent);
    }

    // The same label is clicked twice in a row
    else if (currentElement.textContent == element.textContent){    
        currentElement.className = "Label";
        currentElement = null;
        console.log("New Label: none");
        addLine("none");
    }

    // another label is clicked but there is a current label active
    else {
        currentElement.className = "Label";
        element.className = "LabelKlicked";
        currentElement = element; 
        console.log("New Label: " + element.textContent);   
        addLine(element.textContent);    
    }    
}

/**
 * Checking the participant ID and trigger the redirection to the labeling screen if PID is correct (enough).
 */
function CheckPID() {
    participant = document.getElementById("PID").value;

    if (participant == "") {
        document.getElementById("PIDError").style.display="block";
        return;
    }

    localStorage.setItem("storagePID", participant);

    //document.getElementById("LID").style.display = "grid"; //Only needed using the eel script with starting the cameras
    changeToLabelingScreen();
}

/**
 * Stops the labeling and triggers the download of the log file.
 */
function StopLabeling() {
    changeContent();
    downloadCSV();
}


/**
 * Adds a new line to the log with timestamp (YYYY.MM.DD_HH:mm:ss.uu)
 * 
 * @param {any} label the label that is chosen
 */
function addLine(label){
    date = new Date();
    data += date.getFullYear() + "-" + date.getDate() + "-" + (date.getMonth()+1) + " " + date.getHours() + ":" + date.getMinutes() + ":" + date.getSeconds() + "." + date.getMilliseconds() + ";" + label + "\n"
    dataHTML += date.getFullYear() + "-" + date.getDate() + "-" + (date.getMonth()+1) + " " + date.getHours() + ":" + date.getMinutes() + ":" + date.getSeconds() + "." + date.getMilliseconds() + ";" + label + "<br/>"
    document.getElementById('log').innerHTML = dataHTML;

    var elem = document.getElementById('log');
    elem.scrollTop = elem.scrollHeight;
}

/**
 * Changes the content of the logger. "Replace" <br/> with \n
 */
function changeContent(){
    document.getElementById('log').innerHTML = data;
}


/**
 * Navigates from the start screen to the labeling screen.
 */
function changeToLabelingScreen() {
    window.location.replace("./Labeling.html");
}

/**
 * Changes back to the start screen. 
 */
function changeToStartScreen() {
    currentElement = "";
    localStorage.removeItem("storagePID");
    window.location.replace("./StartScreen.html");
}

/**
 * Adds the participant ID to the screen. 
 */
function setPIDToScreen() {
    document.getElementById("PID_Field").textContent = "Proband " + localStorage.getItem("storagePID");
    document.getElementById("PID_Field").style.backgroundColor = "rgba(0, 0, 0, 0.5)";
    document.getElementById("PID_Field").style.color = "yellow";
}

/*
 * Function to trigger the CSV download
 */
function downloadCSV() {
    data = document.getElementById("log").textContent;
    PID = localStorage.getItem("storagePID");
    filename = PID + '_Labeling.csv';
    const csvContent = `data:text/csv;charset=utf-8,${encodeURIComponent(data)}`;
    const link = document.createElement('a');
    link.setAttribute('href', csvContent);
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    finishedDownload();
}

/**
 * This method is called whenever the download finishes. 
 * The method informs the user that it is safe to close the window or return to the start screen.
 */
function finishedDownload() {
    document.getElementById("log").style.color = "green";
    document.getElementById("log").textContent = "You can close the window/tab now or go back to start. I'm done :)";
    
    changeStopToReset();
}

/*
 * changes the stop button to a return button an changes its style.
 */
function changeStopToReset() {
    document.getElementById("stopButton").style.backgroundColor = "green";
    document.getElementById("stopButton").textContent = "Back to Start";
    document.getElementById("stopButton").setAttribute("onClick", "changeToStartScreen()");
}



