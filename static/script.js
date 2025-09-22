document.addEventListener("DOMContentLoaded", function () {
    let fileInput = document.getElementById("fileInput");
    let fileNameDisplay = document.getElementById("fileNameDisplay");
    let video = document.getElementById("cameraFeed");
    let captureButton = document.getElementById("captureButton");
    let analyzeButton = document.getElementById("analyzeButton");
    let canvas = document.createElement("canvas");
    let imagePreview = document.getElementById("imagePreview");
    let imagePreviewContainer = document.getElementById("image-preview-container");
    let resultContainer = document.getElementById("result");
    let stream = null;

    function clearPreviousResults() {
        resultContainer.innerHTML = "";
    }

    function clearPreviousImage() {
        imagePreview.src = "";
        imagePreview.style.display = "none";
        imagePreviewContainer.style.display = "none";
        fileInput.value = "";
        fileNameDisplay.textContent = "No file chosen";
        analyzeButton.style.display = "none";
    }

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            clearPreviousResults();
            fileNameDisplay.textContent = fileInput.files[0].name;
            analyzeButton.style.display = "inline-block";

            let reader = new FileReader();
            reader.onload = function (e) {
                showImagePreview(e.target.result);
            };
            reader.readAsDataURL(fileInput.files[0]);
        } else {
            fileNameDisplay.textContent = "No file chosen";
            analyzeButton.style.display = "none";
        }
    });

    window.startCamera = function () {
        clearPreviousResults();
        clearPreviousImage();
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (cameraStream) {
                stream = cameraStream;
                video.srcObject = stream;
                video.style.display = "block";
                captureButton.style.display = "inline-block";
            })
            .catch(function (error) {
                console.error("Error accessing camera:", error);
                alert("Failed to access camera. Please check permissions.");
            });
    };

    window.capturePhoto = function () {
        if (!stream) {
            alert("Camera not started!");
            return;
        }
        clearPreviousResults();

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        let context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(function (blob) {
            let formData = new FormData();
            formData.append("file", blob, "captured_image.jpg");

            let reader = new FileReader();
            reader.onload = function (e) {
                showImagePreview(e.target.result);
            };
            reader.readAsDataURL(blob);

            sendImage(formData);
            stopCamera();
        }, "image/jpeg");
    };

    window.uploadImage = function () {
        if (fileInput.files.length === 0) {
            alert("Please select an image.");
            return;
        }
        clearPreviousResults();
        let formData = new FormData();
        formData.append("file", fileInput.files[0]);
        sendImage(formData);
    };

    function showImagePreview(imageSrc) {
        imagePreview.src = imageSrc;
        imagePreview.style.display = "block";
        imagePreviewContainer.style.display = "block";
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.style.display = "none";
            captureButton.style.display = "none";
        }
    }

    function sendImage(formData) {
        fetch("/", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(displayResults)
        .catch(error => console.error("Error:", error));
    }

    function displayResults(data) {
        resultContainer.innerHTML = "<h2>Results:</h2>";

        if (data.humans.length > 0) {
            let humanResults = document.createElement("div");
            humanResults.classList.add("result-grid");

            data.humans.forEach((human, index) => {
                let personCard = document.createElement("div");
                personCard.classList.add("result-card");

                // Convert "Indicates" into bullet points
                let indicatesFormatted = human.risk.includes(" - ") ?
                    human.risk.split(" - ").map(point => `<li>${point}</li>`).join("") :
                    `<li>${human.risk}</li>`;

                personCard.innerHTML = `
                    <h3>Person ${index + 1}</h3>
                    <p><strong>Age:</strong> ${human.age}</p>
                    <p><strong>Gender:</strong> ${human.gender}</p>
                    <p><strong>Height:</strong> ${human.height} cm</p>
                    <p><strong>Weight:</strong> ${human.weight} kg</p>
                    <p><strong>BMI:</strong> ${human.bmi}</p>
                    <p><strong>Indicates:</strong></p>
                    <ul>${indicatesFormatted}</ul>
                `;

                humanResults.appendChild(personCard);
            });

            resultContainer.appendChild(humanResults);
        } else {
            resultContainer.innerHTML += "<p>No human detected.</p>";
        }

        let filteredObjects = data.objects.filter(obj => obj.label.toLowerCase() !== "person");

        if (filteredObjects.length > 0) {
            let objectResults = document.createElement("div");
            objectResults.classList.add("result-grid");

            filteredObjects.forEach((obj, index) => {
                let objectCard = document.createElement("div");
                objectCard.classList.add("result-card");

                objectCard.innerHTML = `
                    <h3>Object ${index + 1}: ${obj.label}</h3>
                    <p><strong>Length:</strong> ${obj.length} cm</p>
                    <p><strong>Width:</strong> ${obj.width} cm</p>
                `;

                objectResults.appendChild(objectCard);
            });

            resultContainer.appendChild(objectResults);
        } else {
            resultContainer.innerHTML += "<p>No objects detected.</p>";
        }
    }
});
