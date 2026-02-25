// Base URL of your FastAPI backend.
// If your backend runs on another host/port, change this value.
const API_BASE_URL = "http://127.0.0.1:8000";

// Grabbing DOM elements we will interact with
const fileInput = document.getElementById("fileInput");
const fileNameDisplay = document.getElementById("fileNameDisplay");
const uploadPreviewImage = document.getElementById("uploadPreviewImage");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const processButton = document.getElementById("processButton");
const statusMessage = document.getElementById("statusMessage");

const loaderOverlay = document.getElementById("loaderOverlay");
const resultImage = document.getElementById("resultImage");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const timestampLabel = document.getElementById("timestampLabel");

const totalCountEl = document.getElementById("totalCount");
const matchCountEl = document.getElementById("matchCount");
const mismatchCountEl = document.getElementById("mismatchCount");

const vehicleTableBody = document.getElementById("vehicleTableBody");

const mismatchAlert = document.getElementById("mismatchAlert");
const viewDetailsLink = document.getElementById("viewDetailsLink");

const mismatchModalBackdrop = document.getElementById("mismatchModalBackdrop");
const modalCloseButton = document.getElementById("modalCloseButton");
const modalOkButton = document.getElementById("modalOkButton");
const modalViewTableButton = document.getElementById("modalViewTableButton");

let selectedFile = null;

// Show / hide loader overlay and update status text
function setLoading(isLoading) {
  if (isLoading) {
    loaderOverlay.style.display = "flex";
    processButton.disabled = true;
    processButton.classList.add("disabled");
    statusMessage.textContent = "Processing image… please wait.";
    statusMessage.classList.remove("error");
  } else {
    loaderOverlay.style.display = "none";
    processButton.disabled = !selectedFile;
    processButton.classList.toggle("disabled", !selectedFile);
    if (!selectedFile) {
      statusMessage.textContent = "Waiting for image upload.";
    }
  }
}

// Reset table to the initial "empty" row
function resetTable() {
  vehicleTableBody.innerHTML =
    '<tr>' +
    '<td colspan="7" class="empty-row">' +
    "No vehicles detected yet. Run analysis to populate this table." +
    "</td>" +
    "</tr>";
}

// Scroll smoothly to the vehicle table
function scrollToTable() {
  const detailsCard = vehicleTableBody.closest(".card");
  if (detailsCard && detailsCard.scrollIntoView) {
    detailsCard.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

// Open mismatch modal
function openMismatchModal() {
  mismatchModalBackdrop.style.display = "flex";
  mismatchModalBackdrop.setAttribute("aria-hidden", "false");
}

// Close mismatch modal
function closeMismatchModal() {
  mismatchModalBackdrop.style.display = "none";
  mismatchModalBackdrop.setAttribute("aria-hidden", "true");
}

// Handle file selection and preview
fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];

  if (!file) {
    selectedFile = null;
    fileNameDisplay.textContent = "No file selected";
    uploadPreviewImage.style.display = "none";
    uploadPlaceholder.style.display = "block";
    processButton.disabled = true;
    processButton.classList.add("disabled");
    statusMessage.textContent = "Waiting for image upload.";
    statusMessage.classList.remove("error");
    return;
  }

  selectedFile = file;
  fileNameDisplay.textContent = file.name;

  // Show a thumbnail preview of the image
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadPreviewImage.src = e.target.result;
    uploadPreviewImage.style.display = "block";
    uploadPlaceholder.style.display = "none";
  };
  reader.readAsDataURL(file);

  processButton.disabled = false;
  processButton.classList.remove("disabled");
  statusMessage.textContent = 'Ready to process. Click "Process Image".';
  statusMessage.classList.remove("error");
});

// Main handler for "Process Image" button
processButton.addEventListener("click", async () => {
  if (!selectedFile) {
    statusMessage.textContent = "Please select an image before processing.";
    statusMessage.classList.add("error");
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  setLoading(true);
  mismatchAlert.style.display = "none";
  closeMismatchModal();

  try {
    // Send the image to FastAPI `/detect` endpoint
    const response = await fetch(API_BASE_URL + "/detect", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Backend returned status " + response.status);
    }

    // Parse JSON body from FastAPI
    const data = await response.json();
    console.log("API response:", data);

    // Safely read the expected fields from the JSON
    const summary =
      data && typeof data.summary === "object" && data.summary !== null
        ? data.summary
        : {};

    // Prefer processed_image_url, fallback to image_url (relative paths from backend)
    var processedImageUrl = "";
    if (data && typeof data.processed_image_url === "string") {
      processedImageUrl = data.processed_image_url;
    } else if (data && typeof data.image_url === "string") {
      processedImageUrl = data.image_url;
    }

    // Read counts from either `summary` or top-level fields for compatibility
    let total = 0;
    if (summary.total_vehicles != null) {
      total = summary.total_vehicles;
    } else if (typeof data.total === "number") {
      total = data.total;
    }

    let matches = 0;
    if (summary.match_count != null) {
      matches = summary.match_count;
    } else if (typeof data.match === "number") {
      matches = data.match;
    }

    let mismatches = 0;
    if (summary.mismatch_count != null) {
      mismatches = summary.mismatch_count;
    } else if (typeof data.mismatch === "number") {
      mismatches = data.mismatch;
    }

    const vehicles = Array.isArray(summary.vehicles)
      ? summary.vehicles
      : [];

    totalCountEl.textContent = total;
    matchCountEl.textContent = matches;
    mismatchCountEl.textContent = mismatches;

    if (processedImageUrl) {
      // Always build full URL using backend base + relative image path
      var imageSrc = API_BASE_URL + processedImageUrl;
      resultImage.src = imageSrc;
      resultImage.style.display = "block";
      resultPlaceholder.style.display = "none";
    } else {
      resultImage.style.display = "none";
      resultPlaceholder.style.display = "block";
    }

    const now = new Date();
    const timeStampText = now.toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
    timestampLabel.textContent = "Last scan: " + timeStampText;

    if (vehicles.length === 0) {
      resetTable();
    } else {
      vehicleTableBody.innerHTML = "";
      vehicles.forEach((vehicle, index) => {
        const row = document.createElement("tr");

        const id = vehicle.id != null ? vehicle.id : index + 1;
        const plate = vehicle.plate != null ? vehicle.plate : "—";
        const status = (vehicle.status || "UNKNOWN").toUpperCase();
        const type = vehicle.type != null ? vehicle.type : "—";
        const color = vehicle.color != null ? vehicle.color : "—";
        const confidence =
          typeof vehicle.confidence === "number"
            ? vehicle.confidence
            : null;
        const time =
          vehicle.time != null ? vehicle.time : timeStampText;

        const idCell = document.createElement("td");
        idCell.textContent = id;
        row.appendChild(idCell);

        const plateCell = document.createElement("td");
        plateCell.textContent = plate;
        row.appendChild(plateCell);

        const statusCell = document.createElement("td");
        const statusSpan = document.createElement("span");
        statusSpan.classList.add("status-pill");
        if (status === "MATCH") {
          statusSpan.classList.add("match");
        } else if (status === "MISMATCH") {
          statusSpan.classList.add("mismatch");
        }
        statusSpan.textContent = status;
        statusCell.appendChild(statusSpan);
        row.appendChild(statusCell);

        const typeCell = document.createElement("td");
        typeCell.textContent = type;
        row.appendChild(typeCell);

        const colorCell = document.createElement("td");
        colorCell.textContent = color;
        row.appendChild(colorCell);

        const confCell = document.createElement("td");
        if (confidence !== null) {
          const pct = Math.round(confidence * 100);
          const label = document.createElement("div");
          label.textContent = pct + "%";
          label.style.fontSize = "0.7rem";

          const bar = document.createElement("div");
          bar.classList.add("confidence-bar");
          const fill = document.createElement("div");
          fill.classList.add("confidence-fill");
          fill.style.width = Math.min(Math.max(pct, 0), 100) + "%";
          bar.appendChild(fill);

          confCell.appendChild(label);
          confCell.appendChild(bar);
        } else {
          confCell.textContent = "—";
        }
        row.appendChild(confCell);

        const timeCell = document.createElement("td");
        timeCell.textContent = time;
        row.appendChild(timeCell);

        vehicleTableBody.appendChild(row);
      });
    }

    if (mismatches > 0) {
      mismatchAlert.style.display = "flex";
      openMismatchModal();
    } else {
      mismatchAlert.style.display = "none";
      closeMismatchModal();
    }

    statusMessage.textContent = "Analysis completed successfully.";
    statusMessage.classList.remove("error");
  } catch (error) {
    console.error(error);
    statusMessage.textContent =
      "Failed to process image. Please check the backend or try again.";
    statusMessage.classList.add("error");
  } finally {
    setLoading(false);
  }
});

// Small banner "View flagged vehicles" link
viewDetailsLink.addEventListener("click", () => {
  openMismatchModal();
  scrollToTable();
});

// Modal buttons
modalCloseButton.addEventListener("click", closeMismatchModal);
modalOkButton.addEventListener("click", closeMismatchModal);
modalViewTableButton.addEventListener("click", () => {
  closeMismatchModal();
  scrollToTable();
});

// Close modal when clicking outside the dialog
mismatchModalBackdrop.addEventListener("click", (event) => {
  if (event.target === mismatchModalBackdrop) {
    closeMismatchModal();
  }
});

// Make the custom "Choose Image" button trigger the hidden file input
const fauxChooseButton = document.querySelector(".file-input .btn.secondary");
if (fauxChooseButton) {
  fauxChooseButton.addEventListener("click", () => {
    fileInput.click();
  });
}

