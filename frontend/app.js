(function () {
  "use strict";

  var API_URL = (window.FRUIT_API_URL || "").replace(/\/+$/, "");

  var dropZone = document.getElementById("dropZone");
  var dropZoneContent = document.getElementById("dropZoneContent");
  var fileInput = document.getElementById("fileInput");
  var browseBtn = document.getElementById("browseBtn");
  var cameraBtn = document.getElementById("cameraBtn");
  var classifyBtn = document.getElementById("classifyBtn");
  var preview = document.getElementById("preview");
  var fileInfo = document.getElementById("fileInfo");
  var loading = document.getElementById("loading");
  var resultCard = document.getElementById("resultCard");
  var fruitName = document.getElementById("fruitName");
  var confidence = document.getElementById("confidence");
  var confidenceBar = document.getElementById("confidenceBar");
  var topPredictions = document.getElementById("topPredictions");
  var errorBox = document.getElementById("errorBox");
  var cameraModal = document.getElementById("cameraModal");
  var cameraVideo = document.getElementById("cameraVideo");
  var captureBtn = document.getElementById("captureBtn");
  var closeCameraBtn = document.getElementById("closeCameraBtn");

  var selectedFile = null;
  var cameraStream = null;

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  }

  function clearError() {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
  }

  function hideResult() {
    resultCard.classList.add("hidden");
  }

  function setLoading(on) {
    loading.classList.toggle("hidden", !on);
    classifyBtn.disabled = on;
    classifyBtn.textContent = on ? "Classifying..." : "Classify Fruit";
  }

  function handleFile(file) {
    clearError();
    hideResult();
    if (!file) {
      return;
    }
    if (!file.type || file.type.indexOf("image/") !== 0) {
      showError("Please choose an image file (PNG, JPG, WebP...).");
      return;
    }
    if (file.size === 0) {
      showError("The selected file is empty.");
      return;
    }
    selectedFile = file;
    var objectUrl = URL.createObjectURL(file);
    preview.addEventListener(
      "load",
      function () {
        if (preview.getAttribute("data-src") === objectUrl) {
          URL.revokeObjectURL(objectUrl);
        }
      },
      { once: true }
    );
    preview.setAttribute("data-src", objectUrl);
    preview.src = objectUrl;
    preview.classList.remove("hidden");
    dropZoneContent.classList.add("hidden");
    classifyBtn.disabled = false;
    fileInfo.textContent =
      file.name + " (" + formatBytes(file.size) + ")";
  }

  function renderResult(data) {
    fruitName.textContent = data.predicted_class;
    confidence.textContent = (data.confidence * 100).toFixed(1) + "%";
    confidenceBar.style.width =
      Math.min(100, Math.max(0, data.confidence * 100)) + "%";
    topPredictions.innerHTML = "";
    data.top_predictions.forEach(function (pred, index) {
      var li = document.createElement("li");

      var rank = document.createElement("span");
      rank.className = "rank";
      rank.textContent = index + 1 + ".";

      var name = document.createElement("span");
      name.className = "name";
      name.textContent = pred.class;

      var pct = document.createElement("span");
      pct.className = "pct";
      pct.textContent = (pred.confidence * 100).toFixed(1) + "%";

      li.appendChild(rank);
      li.appendChild(name);
      li.appendChild(pct);
      topPredictions.appendChild(li);
    });
    resultCard.classList.remove("hidden");
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function classify() {
    if (!selectedFile) {
      return;
    }
    clearError();
    hideResult();
    setLoading(true);

    var form = new FormData();
    form.append("image", selectedFile, selectedFile.name);

    fetch(API_URL + "/api/predict", {
      method: "POST",
      body: form,
    })
      .then(function (response) {
        if (!response.ok) {
          return response
            .json()
            .catch(function () {
              return { detail: response.status + " " + response.statusText };
            })
            .then(function (data) {
              throw new Error(data.detail || "Prediction failed");
            });
        }
        return response.json();
      })
      .then(function (data) {
        renderResult(data);
      })
      .catch(function (err) {
        if (err instanceof TypeError) {
          showError(
            "Cannot reach the classification service. " +
              "Check that the backend is running and try again."
          );
        } else {
          showError(err.message || "Prediction failed. Please try again.");
        }
      })
      .finally(function () {
        setLoading(false);
      });
  }

  // --- Upload events ---

  browseBtn.addEventListener("click", function (event) {
    event.stopPropagation();
    fileInput.click();
  });

  dropZone.addEventListener("click", function () {
    fileInput.click();
  });

  dropZone.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", function () {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
    fileInput.value = "";
  });

  ["dragenter", "dragover"].forEach(function (eventName) {
    dropZone.addEventListener(eventName, function (event) {
      event.preventDefault();
      event.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(function (eventName) {
    dropZone.addEventListener(eventName, function (event) {
      event.preventDefault();
      event.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", function (event) {
    var files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  });

  classifyBtn.addEventListener("click", classify);

  // --- Camera capture ---

  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    cameraBtn.classList.remove("hidden");
  }

  cameraBtn.addEventListener("click", function () {
    clearError();
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment" } })
      .then(function (stream) {
        cameraStream = stream;
        cameraVideo.srcObject = stream;
        cameraModal.classList.remove("hidden");
      })
      .catch(function () {
        showError("Camera is not available on this device or browser.");
      });
  });

  function closeCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(function (track) {
        track.stop();
      });
      cameraStream = null;
    }
    cameraVideo.srcObject = null;
    cameraModal.classList.add("hidden");
  }

  captureBtn.addEventListener("click", function () {
    if (!cameraVideo.videoWidth) {
      return;
    }
    var canvas = document.createElement("canvas");
    canvas.width = cameraVideo.videoWidth;
    canvas.height = cameraVideo.videoHeight;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(cameraVideo, 0, 0);
    canvas.toBlob(function (blob) {
      closeCamera();
      if (blob) {
        var file = new File([blob], "camera-capture.jpg", {
          type: "image/jpeg",
        });
        handleFile(file);
      }
    }, "image/jpeg");
  });

  closeCameraBtn.addEventListener("click", closeCamera);

  cameraModal.addEventListener("click", function (event) {
    if (event.target === cameraModal) {
      closeCamera();
    }
  });
})();
