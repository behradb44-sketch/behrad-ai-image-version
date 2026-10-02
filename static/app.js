"use strict";

const $ = (selector) => document.querySelector(selector);

const promptInput = $("#prompt");
const counter = $("#counter");

const modelSelect = $("#model");
const sizeSelect = $("#size");

const enhanceInput = $("#enhance");
const safeInput = $("#safe");

const seedInput = $("#seed");

const generateButton = $("#generate");
const randomSeedButton = $("#random-seed");

const preview = $("#preview");

const emptyState = $("#empty-state");
const loadingState = $("#loading-state");
const resultImage = $("#result-image");

const actions = $("#actions");

const downloadButton = $("#download");
const regenerateButton = $("#regenerate");
const copyPromptButton = $("#copy-prompt");

const statusBox = $("#status");
const modelStatus = $("#model-status");

const historyContainer = $("#history");
const historyEmpty = $("#history-empty");
const clearHistoryButton = $("#clear-history");

const quickButtons = document.querySelectorAll(
    ".quick-prompts button"
);


const HISTORY_KEY = "behrad_image_ai_history_v1";
const MAX_HISTORY = 8;

let lastGeneration = null;
let currentImageData = null;


// ============================================================
// UI helpers
// ============================================================

function setStatus(message = "", type = "") {
    statusBox.textContent = message;
    statusBox.className = "status";

    if (type) {
        statusBox.classList.add(type);
    }
}


function setBusy(busy) {
    generateButton.disabled = busy;
    randomSeedButton.disabled = busy;

    if (busy) {
        generateButton.classList.add("loading");

        generateButton.querySelector("span:nth-child(2)").textContent =
            "در حال ساخت...";

        modelStatus.textContent = "GENERATING";
    } else {
        generateButton.classList.remove("loading");

        generateButton.querySelector("span:nth-child(2)").textContent =
            "ساخت تصویر";

        modelStatus.textContent = "READY";
    }
}


function showEmpty() {
    emptyState.classList.remove("hidden");
    loadingState.classList.add("hidden");
    resultImage.classList.add("hidden");
    actions.classList.add("hidden");
}


function showLoading() {
    emptyState.classList.add("hidden");
    loadingState.classList.remove("hidden");
    resultImage.classList.add("hidden");
    actions.classList.add("hidden");
}


function showResult(src) {
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");

    resultImage.src = src;
    resultImage.classList.remove("hidden");

    actions.classList.remove("hidden");
}


// ============================================================
// Prompt
// ============================================================

function updateCounter() {
    counter.textContent = `${promptInput.value.length} / 4000`;
}

promptInput.addEventListener("input", updateCounter);

updateCounter();


quickButtons.forEach((button) => {
    button.addEventListener("click", () => {
        promptInput.value = button.dataset.prompt || "";

        updateCounter();

        promptInput.focus();
    });
});


promptInput.addEventListener("keydown", (event) => {
    if (
        event.key === "Enter" &&
        (event.ctrlKey || event.metaKey)
    ) {
        event.preventDefault();

        generateImage();
    }
});


// ============================================================
// Size
// ============================================================

function getDimensions() {
    const value = sizeSelect.value;

    const parts = value.split("x");

    if (parts.length !== 2) {
        return {
            width: 1024,
            height: 1024,
        };
    }

    return {
        width: Number(parts[0]),
        height: Number(parts[1]),
    };
}


// ============================================================
// Seed
// ============================================================

function randomSeed() {
    const seed = Math.floor(
        Math.random() * 2147483647
    );

    seedInput.value = seed;
}

randomSeedButton.addEventListener(
    "click",
    randomSeed
);


// ============================================================
// History
// ============================================================

function getHistory() {
    try {
        const raw = localStorage.getItem(HISTORY_KEY);

        if (!raw) {
            return [];
        }

        const parsed = JSON.parse(raw);

        return Array.isArray(parsed)
            ? parsed
            : [];
    } catch {
        return [];
    }
}


function saveHistory(items) {
    try {
        localStorage.setItem(
            HISTORY_KEY,
            JSON.stringify(items.slice(0, MAX_HISTORY))
        );
    } catch {
        // localStorage may be unavailable/full.
    }
}


function addHistory(item) {
    const items = getHistory();

    items.unshift(item);

    saveHistory(items);

    renderHistory();
}


function renderHistory() {
    const items = getHistory();

    historyContainer.innerHTML = "";

    if (!items.length) {
        historyEmpty.classList.remove("hidden");
        return;
    }

    historyEmpty.classList.add("hidden");

    items.forEach((item, index) => {
        const card = document.createElement("button");

        card.type = "button";
        card.className = "history-card";

        card.title = item.prompt || "Generated image";

        const image = document.createElement("img");

        image.src = item.image;
        image.alt = "History image";
        image.loading = "lazy";

        const overlay = document.createElement("div");

        overlay.className = "history-overlay";

        const text = document.createElement("span");

        text.textContent = item.prompt || "تصویر";

        overlay.appendChild(text);

        card.appendChild(image);
        card.appendChild(overlay);

        card.addEventListener("click", () => {
            currentImageData = item.image;

            lastGeneration = {
                prompt: item.prompt,
                model: item.model,
                width: item.width,
                height: item.height,
                seed: item.seed,
                enhance: item.enhance,
                safe: item.safe,
            };

            promptInput.value = item.prompt || "";

            if (item.model) {
                modelSelect.value = item.model;
            }

            if (
                item.width &&
                item.height
            ) {
                const wanted =
                    `${item.width}x${item.height}`;

                const option =
                    [...sizeSelect.options]
                        .find(
                            (opt) => opt.value === wanted
                        );

                if (option) {
                    sizeSelect.value = wanted;
                }
            }

            seedInput.value =
                Number.isInteger(item.seed)
                    ? item.seed
                    : -1;

            enhanceInput.checked =
                Boolean(item.enhance);

            safeInput.checked =
                item.safe !== false;

            promptInput.value =
                item.prompt || "";

            updateCounter();

            showResult(item.image);

            window.scrollTo({
                top: preview.getBoundingClientRect().top
                    + window.scrollY
                    - 30,
                behavior: "smooth",
            });
        });

        historyContainer.appendChild(card);
    });
}


clearHistoryButton.addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);

    renderHistory();

    setStatus(
        "History پاک شد.",
        "success"
    );
});


// ============================================================
// Models
// ============================================================

async function loadModels() {
    try {
        const response = await fetch(
            "/api/models",
            {
                headers: {
                    "Accept": "application/json",
                },
            }
        );

        if (!response.ok) {
            return;
        }

        const data = await response.json();

        if (
            !data ||
            !Array.isArray(data.models) ||
            !data.models.length
        ) {
            return;
        }

        const current = modelSelect.value;

        modelSelect.innerHTML = "";

        data.models.forEach((model) => {
            if (!model || !model.id) {
                return;
            }

            const option =
                document.createElement("option");

            option.value = model.id;

            option.textContent =
                model.name || model.id;

            modelSelect.appendChild(option);
        });

        const hasCurrent =
            [...modelSelect.options]
                .some(
                    (option) =>
                        option.value === current
                );

        if (hasCurrent) {
            modelSelect.value = current;
        }

    } catch {
        // Static fallback models remain available.
    }
}


// ============================================================
// Generate
// ============================================================

async function generateImage() {
    const prompt = promptInput.value.trim();

    if (prompt.length < 2) {
        setStatus(
            "اول یک Prompt بنویس.",
            "error"
        );

        promptInput.focus();

        return;
    }

    if (prompt.length > 4000) {
        setStatus(
            "Prompt بیش از حد طولانی است.",
            "error"
        );

        return;
    }

    const dimensions = getDimensions();

    const seed = Number(
        seedInput.value
    );

    if (
        !Number.isInteger(seed) ||
        seed < -1 ||
        seed > 2147483647
    ) {
        setStatus(
            "Seed نامعتبر است.",
            "error"
        );

        return;
    }

    const payload = {
        prompt,
        model: modelSelect.value,
        width: dimensions.width,
        height: dimensions.height,
        seed,
        enhance: enhanceInput.checked,
        safe: safeInput.checked,
    };

    lastGeneration = payload;

    setBusy(true);
    showLoading();

    setStatus(
        "در حال ارتباط با موتور تولید تصویر...",
        "loading"
    );

    try {
        const response = await fetch(
            "/api/generate",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json",

                    "Accept":
                        "application/json",
                },

                body: JSON.stringify(payload),
            }
        );

        let data = null;

        try {
            data = await response.json();
        } catch {
            data = null;
        }

        if (!response.ok) {
            throw new Error(
                data?.detail ||
                "تولید تصویر ناموفق بود."
            );
        }

        if (
            !data ||
            !data.image ||
            typeof data.image !== "string"
        ) {
            throw new Error(
                "پاسخ تصویر معتبر نبود."
            );
        }

        currentImageData = data.image;

        showResult(currentImageData);

        addHistory({
            image: currentImageData,
            prompt: payload.prompt,
            model: payload.model,
            width: payload.width,
            height: payload.height,
            seed: payload.seed,
            enhance: payload.enhance,
            safe: payload.safe,
            createdAt: Date.now(),
        });

        setStatus(
            "تصویر با موفقیت ساخته شد ✨",
            "success"
        );

    } catch (error) {
        console.error(error);

        showEmpty();

        setStatus(
            error?.message ||
            "خطایی در ساخت تصویر رخ داد.",
            "error"
        );

    } finally {
        setBusy(false);
    }
}


// ============================================================
// Buttons
// ============================================================

generateButton.addEventListener(
    "click",
    generateImage
);


regenerateButton.addEventListener(
    "click",
    () => {
        if (!lastGeneration) {
            return;
        }

        promptInput.value =
            lastGeneration.prompt || "";

        modelSelect.value =
            lastGeneration.model ||
            modelSelect.value;

        seedInput.value =
            Number.isInteger(lastGeneration.seed)
                ? lastGeneration.seed
                : -1;

        enhanceInput.checked =
            Boolean(lastGeneration.enhance);

        safeInput.checked =
            lastGeneration.safe !== false;

        updateCounter();

        generateImage();
    }
);


downloadButton.addEventListener(
    "click",
    () => {
        if (!currentImageData) {
            return;
        }

        const link =
            document.createElement("a");

        link.href = currentImageData;

        link.download =
            `behrad-image-${Date.now()}.png`;

        document.body.appendChild(link);

        link.click();

        link.remove();
    }
);


copyPromptButton.addEventListener(
    "click",
    async () => {
        const prompt =
            lastGeneration?.prompt ||
            promptInput.value.trim();

        if (!prompt) {
            return;
        }

        try {
            await navigator.clipboard.writeText(
                prompt
            );

            setStatus(
                "Prompt کپی شد 📋",
                "success"
            );

        } catch {
            setStatus(
                "کپی کردن Prompt ممکن نبود.",
                "error"
            );
        }
    }
);


// ============================================================
// Startup
// ============================================================

showEmpty();

renderHistory();

loadModels();
