particlesJS.load("particles-js", "/static/particles.json", function () {
  console.log("particles.js loaded");
});

let uploadedFileName = null;

document.getElementById("file-input").addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      document.getElementById(
        "image-preview"
      ).innerHTML = `<img src="${e.target.result}" class="img-fluid rounded" style="max-height: 300px;">`;
      uploadFile(file);
    };
    reader.readAsDataURL(file);
  }
});

function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/upload", true);

  xhr.upload.onprogress = function (e) {
    if (e.lengthComputable) {
      const percentComplete = (e.loaded / e.total) * 100;
      document.querySelector(".progress-bar").style.width =
        percentComplete + "%";
      document
        .querySelector(".progress-bar")
        .setAttribute("aria-valuenow", percentComplete);
    }
  };

  xhr.onload = function () {
    if (xhr.status === 200) {
      const response = JSON.parse(xhr.responseText);
      if (response.success) {
        document.getElementById("upload-status").textContent =
          "آپلود با موفقیت انجام شد.";
        uploadedFileName = response.filename;
        checkFormCompletion();
      } else {
        document.getElementById("upload-status").textContent =
          "خطا در آپلود: " + response.message;
      }
    }
  };

  xhr.send(formData);
  document.querySelector(".progress").style.display = "block";
}

document.querySelectorAll(".symptom-tag").forEach((tag) => {
  tag.addEventListener("click", function () {
    this.classList.toggle("active");
    checkFormCompletion();
  });
});

document
  .getElementById("diagnosis-form")
  .addEventListener("input", checkFormCompletion);

function checkFormCompletion() {
  const form = document.getElementById("diagnosis-form");
  const requiredFields = form.querySelectorAll("[required]");
  const diagnosisBtn = document.getElementById("diagnose-btn");

  let isFormComplete = true;
  requiredFields.forEach((field) => {
    if (!field.value) {
      isFormComplete = false;
    }
  });

  if (isFormComplete && uploadedFileName) {
    diagnosisBtn.disabled = false;
  } else {
    diagnosisBtn.disabled = true;
  }
}

document.addEventListener("DOMContentLoaded", function () {
  var form = document.getElementById("diagnosis-form");
  var fileInput = document.getElementById("file-input");

  form.addEventListener("submit", function (e) {
    "  e.preventDefault();";

    if (fileInput.files.length === 0) {
      diagnoseBtn.disabled = true;
      alert("لطفاً یک تصویر انتخاب کنید.");
      return;
    }

    var formData = new FormData(form);

    // اطمینان از اضافه شدن فایل به FormData
    formData.append("file", fileInput.files[0]);

    fetch("/diagnose", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json()) // تغییر به json() برای پردازش پاسخ JSON
      .then((data) => {
        if (data.error) {
          alert(data.error);
        } else {
          document.body.innerHTML = data.html;
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("خطا در ارتباط با سرور");
      });
  });
});

document.querySelectorAll(".body-part").forEach((part) => {
  part.addEventListener("click", function () {
    document
      .querySelectorAll(".body-part")
      .forEach((p) => p.classList.remove("active"));
    this.classList.add("active");
    document.querySelector(".body-part-label").textContent = this.dataset.name;
  });
});

document.getElementById("age").addEventListener("input", function () {
  const ageInput = document.getElementById("age");
  const ageError = document.getElementById("age-error");

  if (ageInput.value > 120) {
    ageError.style.display = "inline";
    ageInput.value = 120;
  } else {
    ageError.style.display = "none";
  }
});

document
  .getElementById("diagnosis-form")
  .addEventListener("submit", function (event) {
    const skinType = document.getElementById("skin-type");
    const gender = document.getElementById("gender");
    const ethnicity = document.getElementById("ethnicity");

    let isValid = true;

    if (skinType.value === "") {
      document.getElementById("skin-type-error").style.display = "inline";
      isValid = false;
    } else {
      document.getElementById("skin-type-error").style.display = "none";
    }

    if (gender.value === "") {
      document.getElementById("gender-error").style.display = "inline";
      isValid = false;
    } else {
      document.getElementById("gender-error").style.display = "none";
    }

    if (ethnicity.value === "") {
      document.getElementById("ethnicity-error").style.display = "inline";
      isValid = false;
    } else {
      document.getElementById("ethnicity-error").style.display = "none";
    }

    if (!isValid) {
      event.preventDefault();
    }
  });

const additionalInfo = document.getElementById("additional-info");
const infoError = document.getElementById("info-error");
const remainingChars = document.getElementById("remaining-chars");

additionalInfo.addEventListener("input", function () {
  const maxLength = 150;
  const currentLength = additionalInfo.value.length;
  const remaining = maxLength - currentLength;

  remainingChars.textContent = remaining;

  if (currentLength > maxLength) {
    infoError.style.display = "inline";
  } else {
    infoError.style.display = "none";
  }
});

particlesJS.load("particles-js", "/static/particles.json", function () {
  console.log("particles.js loaded");
});

// Initialize Particles.js
particlesJS("particles-js", {
  particles: {
    number: {
      value: 80,
      density: {
        enable: true,
        value_area: 800,
      },
    },
    color: {
      value: "#007bff",
    },
    shape: {
      type: "circle",
      stroke: {
        width: 0,
        color: "#000000",
      },
      polygon: {
        nb_sides: 5,
      },
    },
    opacity: {
      value: 0.5,
      random: false,
      anim: {
        enable: false,
        speed: 1,
        opacity_min: 0.1,
        sync: false,
      },
    },
    size: {
      value: 3,
      random: true,
      anim: {
        enable: false,
        speed: 40,
        size_min: 0.1,
        sync: false,
      },
    },
    line_linked: {
      enable: true,
      distance: 150,
      color: "#007bff",
      opacity: 0.4,
      width: 1,
    },
    move: {
      enable: true,
      speed: 6,
      direction: "none",
      random: false,
      straight: false,
      out_mode: "out",
      bounce: false,
      attract: {
        enable: false,
        rotateX: 600,
        rotateY: 1200,
      },
    },
  },
  interactivity: {
    detect_on: "canvas",
    events: {
      onhover: {
        enable: true,
        mode: "repulse",
      },
      onclick: {
        enable: true,
        mode: "push",
      },
      resize: true,
    },
    modes: {
      grab: {
        distance: 400,
        line_linked: {
          opacity: 1,
        },
      },
      bubble: {
        distance: 400,
        size: 40,
        duration: 2,
        opacity: 8,
        speed: 3,
      },
      repulse: {
        distance: 200,
        duration: 0.4,
      },
      push: {
        particles_nb: 4,
      },
      remove: {
        particles_nb: 2,
      },
    },
  },
  retina_detect: true,
});

// Typing animation for the main title
var typed = new Typed("#main-title", {
  strings: ["سیستم پیشرفته هوش مصنوعی تشخیص بیماری‌های پوستی"],
  typeSpeed: 50,
  backSpeed: 0,
  loop: false,
  showCursor: false,
  onComplete: function (self) {
    $("#main-title").addClass("animate__animated animate__pulse");
  },
});

$("form").on("submit", function (event) {
  event.preventDefault();
  var formData = new FormData(this);
  formData.append("file", fileInput.files[0]);

  $.ajax({
    url: "/diagnose",
    type: "POST",
    data: formData,
    contentType: false,
    cache: false,
    processData: false,
    success: function (response) {
      if (response.error) {
        alert(response.error);
      } else {
        $("#result").html(response.html);
      }
    },
    error: function () {
      alert("خطا در ارسال درخواست");
    },
  });
});
