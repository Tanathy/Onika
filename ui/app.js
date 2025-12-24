const API_BASE = "/api";

// --- TABS ---
function openTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    // Deactivate all nav items
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    
    // Show target tab
    document.getElementById(`tab-${tabName}`).classList.add('active');
    // Activate target nav item
    document.querySelectorAll(`.nav-item[onclick="openTab('${tabName}')"]`).forEach(el => el.classList.add('active'));

    if (tabName === 'metadata') {
        fetchOutputs();
    }
    if (tabName === 'conversion') {
        refreshConversionFiles();
    }
    if (tabName === 'console') {
        drawChart();
    }
}

function openSubTab(tabName) {
    document.querySelectorAll('.sub-tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.sub-tab-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(tabName).classList.add('active');
    document.querySelectorAll(`.sub-tab-btn[onclick="openSubTab('${tabName}')"]`).forEach(el => el.classList.add('active'));
}

// --- CROSS REFERENCES (XREF) ---
let _xrefLayerEl = null;
let _xrefTemplateEl = null;
let _xrefScrollContainer = null;

let _xrefTicking = false;
let _xrefPointers = [];
let _xrefActiveKeys = new Set();
let _xrefHighlightTimers = new WeakMap();

const XREF_HIGHLIGHT_MS = 5000;
const XREF_POINTER_FADE_MS = 900;

function _cssEscape(s) {
    if (window.CSS && typeof window.CSS.escape === 'function') return window.CSS.escape(s);
    return String(s).replace(/[^a-zA-Z0-9_\-]/g, '_');
}

function _getPointerTargetByKey(key) {
    if (!key) return null;
    const safe = _cssEscape(key);
    const all = Array.from(document.querySelectorAll(`[data-pointer="${safe}"]`));
    if (all.length === 0) return null;
    // Prefer a visible element (important when the same pointer key exists in multiple sub-tabs).
    const visible = all.find(el => el && el.offsetParent !== null);
    return visible || all[0];
}

function _hasActivePointerForKey(key) {
    return _xrefActiveKeys.has(String(key || ''));
}

function _getTabButton(tabName) {
    if (!tabName) return null;
    return document.querySelector(`.nav-item[onclick="openTab('${tabName}')"]`) || document.getElementById(`nav-${tabName}`);
}

function _getSubTabButton(subtabName) {
    if (!subtabName) return null;
    return document.querySelector(`.sub-tab-btn[onclick="openSubTab('${subtabName}')"]`) || document.getElementById(`subtab-${subtabName.replace('cfg-', '')}`);
}

function _isInViewport(rect, pad = 8) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    return (
        rect.bottom > pad &&
        rect.top < (vh - pad) &&
        rect.right > pad &&
        rect.left < (vw - pad)
    );
}

function _clamp(n, min, max) {
    return Math.max(min, Math.min(max, n));
}

function _rectsIntersect(a, b) {
    if (!a || !b) return false;
    return !(a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom);
}

function _pulseXrefTarget(el, durationMs = XREF_HIGHLIGHT_MS) {
    if (!el || !el.classList) return;
    try {
        // Restart animation cleanly if it was already active.
        if (el.classList.contains('xref-target-highlight')) {
            el.classList.remove('xref-target-highlight');
            // Force reflow so the animation restarts.
            void el.offsetWidth;
        }
        el.classList.add('xref-target-highlight');
        const prev = _xrefHighlightTimers.get(el);
        if (prev) clearTimeout(prev);
        const tid = setTimeout(() => {
            try { el.classList.remove('xref-target-highlight'); } catch (_) {}
            _xrefHighlightTimers.delete(el);
        }, durationMs);
        _xrefHighlightTimers.set(el, tid);
    } catch (_) {
        // ignore
    }
}



function _collectAvoidRectsForPointer(extraPadding = 10) {
    const els = [];
    els.push(...document.querySelectorAll('.nav-item'));
    els.push(...document.querySelectorAll('.sub-tab-btn'));

    const rects = [];
    for (const el of els) {
        if (!el || el.offsetParent === null) continue;
        const r = el.getBoundingClientRect?.();
        if (!r) continue;
        if (!_isInViewport(r, 0)) continue;
        rects.push({
            left: r.left - extraPadding,
            right: r.right + extraPadding,
            top: r.top - extraPadding,
            bottom: r.bottom + extraPadding,
        });
    }
    return rects;
}

function _pickNonOverlappingPointerPosition(aimRect, pointerW, pointerH, avoidRects = []) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const pad = 16;
    const gap = 14;
    const avoid = 8;

    const w = Math.max(1, pointerW || 1);
    const h = Math.max(1, pointerH || 1);

    const minX = pad + w / 2;
    const maxX = vw - pad - w / 2;
    const minY = pad + h / 2;
    const maxY = vh - pad - h / 2;

    const cx = aimRect.left + aimRect.width / 2;
    const cy = aimRect.top + aimRect.height / 2;

    const expandedAim = {
        left: aimRect.left - avoid,
        right: aimRect.right + avoid,
        top: aimRect.top - avoid,
        bottom: aimRect.bottom + avoid,
    };

    const spaces = [
        { side: 'right', space: (vw - pad) - aimRect.right },
        { side: 'left', space: aimRect.left - pad },
        { side: 'bottom', space: (vh - pad) - aimRect.bottom },
        { side: 'top', space: aimRect.top - pad },
    ].sort((a, b) => b.space - a.space);

    const candidates = [];
    for (const s of spaces) {
        if (s.side === 'right') {
            candidates.push({
                px: aimRect.right + gap + w / 2,
                py: _clamp(cy, minY, maxY),
            });
        } else if (s.side === 'left') {
            candidates.push({
                px: aimRect.left - gap - w / 2,
                py: _clamp(cy, minY, maxY),
            });
        } else if (s.side === 'bottom') {
            candidates.push({
                px: _clamp(cx, minX, maxX),
                py: aimRect.bottom + gap + h / 2,
            });
        } else if (s.side === 'top') {
            candidates.push({
                px: _clamp(cx, minX, maxX),
                py: aimRect.top - gap - h / 2,
            });
        }
    }

    // Add diagonals as fallbacks for tight spaces.
    candidates.push(
        { px: aimRect.right + gap + w / 2, py: aimRect.top - gap - h / 2 },
        { px: aimRect.right + gap + w / 2, py: aimRect.bottom + gap + h / 2 },
        { px: aimRect.left - gap - w / 2, py: aimRect.top - gap - h / 2 },
        { px: aimRect.left - gap - w / 2, py: aimRect.bottom + gap + h / 2 },
    );

    for (const c of candidates) {
        const px = _clamp(c.px, minX, maxX);
        const py = _clamp(c.py, minY, maxY);

        const pr = {
            left: px - w / 2,
            right: px + w / 2,
            top: py - h / 2,
            bottom: py + h / 2,
        };

        if (_rectsIntersect(pr, expandedAim)) continue;
        let blocked = false;
        for (const ar of avoidRects) {
            if (_rectsIntersect(pr, ar)) {
                blocked = true;
                break;
            }
        }
        if (blocked) continue;
        return { px, py };
    }

    // Worst case: clamp to center, then push away from the aim rect.
    let px = _clamp(cx, minX, maxX);
    let py = _clamp(cy, minY, maxY);

    const pr0 = {
        left: px - w / 2,
        right: px + w / 2,
        top: py - h / 2,
        bottom: py + h / 2,
    };
    if (_rectsIntersect(pr0, expandedAim)) {
        const dxRight = (expandedAim.right - pr0.left);
        const dxLeft = (pr0.right - expandedAim.left);
        const dyDown = (expandedAim.bottom - pr0.top);
        const dyUp = (pr0.bottom - expandedAim.top);

        const pushX = dxRight < dxLeft ? dxRight : -dxLeft;
        const pushY = dyDown < dyUp ? dyDown : -dyUp;

        if (Math.abs(pushX) > Math.abs(pushY)) px = _clamp(px + pushX, minX, maxX);
        else py = _clamp(py + pushY, minY, maxY);
    }

    return { px, py };
}

function _pickAnchorPoint(rect) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const pad = 16;

    const tx = rect.left + rect.width / 2;
    const ty = rect.top + rect.height / 2;
    const inView = _isInViewport(rect, pad);

    let px;
    let py;

    if (inView) {
        const preferLeft = rect.left - 26;
        const preferRight = rect.right + 26;
        if (preferLeft > pad) px = preferLeft;
        else px = Math.min(vw - pad, preferRight);
        py = Math.max(pad, Math.min(vh - pad, ty));
    } else {
        const offTop = rect.bottom < pad;
        const offBottom = rect.top > (vh - pad);
        const offLeft = rect.right < pad;
        const offRight = rect.left > (vw - pad);

        px = Math.max(pad, Math.min(vw - pad, tx));
        py = Math.max(pad, Math.min(vh - pad, ty));
        if (offTop) py = pad;
        else if (offBottom) py = vh - pad;
        if (offLeft) px = pad;
        else if (offRight) px = vw - pad;
    }

    return { px, py, tx, ty };
}

function _updatePointerGeometry(p) {
    const sourceRect = p.sourceEl?.getBoundingClientRect?.();

    const waitingEl = p.waitingEl || (p.waitingSelector ? document.querySelector(p.waitingSelector) : null);
    const waitingRect = waitingEl?.getBoundingClientRect?.();

    const targetEl = _getPointerTargetByKey(p.targetKey);
    const targetRect = targetEl?.getBoundingClientRect?.();
    const targetVisible = !!(targetRect && _isInViewport(targetRect, 8) && targetEl.offsetParent !== null);

    // Stage 1: wait at a specific UI anchor (tab/subtab button) if provided
    // Stage 2: once user opened it (or it becomes visible), move to the real target.
    let rect = null;
    let aimRect = null;

    if (p.stage === 1) {
        if (waitingRect) {
            rect = waitingRect;
            aimRect = waitingRect;
        } else if (sourceRect) {
            rect = sourceRect;
            aimRect = sourceRect;
        }

        // If somehow target is already visible, allow stage-2 travel immediately.
        if (targetVisible) {
            p.stage = 2;
        }
    }

    if (p.stage === 2) {
        if (targetVisible) {
            rect = targetRect;
            aimRect = targetRect;
            p.mode = 'to-target';
            p.targetEl = targetEl;
        } else {
            // If target isn't visible yet, keep waiting at the waiting anchor.
            if (waitingRect) {
                rect = waitingRect;
                aimRect = waitingRect;
            } else if (sourceRect) {
                rect = sourceRect;
                aimRect = sourceRect;
            }
            p.mode = 'waiting';
            p.targetEl = null;
        }
    }

    if (!rect) {
        p.dead = true;
        return;
    }

    // Place the pointer as a label near the aim rect without covering it.
    const w = p.el?.offsetWidth || 180;
    const h = p.el?.offsetHeight || 34;
    const avoidRects = _collectAvoidRectsForPointer(10);
    const pos = _pickNonOverlappingPointerPosition(aimRect || rect, w, h, avoidRects);
    p.desiredX = pos.px;
    p.desiredY = pos.py;
}

function _tickXrefPointers() {
    _xrefTicking = false;
    if (_xrefPointers.length === 0) return;

    for (const p of _xrefPointers) {
        if (p.dead) continue;
        _updatePointerGeometry(p);

        const lerp = 0.18;
        p.x = p.x + (p.desiredX - p.x) * lerp;
        p.y = p.y + (p.desiredY - p.y) * lerp;

        p.el.style.left = `${p.x}px`;
        p.el.style.top = `${p.y}px`;

        // If we reached our non-overlapping target position, fade out.
        if (p.mode === 'to-target' && p.targetEl) {
            const trect = p.targetEl.getBoundingClientRect();
            if (_isInViewport(trect, 8)) {
                const distToDesired = Math.hypot(p.desiredX - p.x, p.desiredY - p.y);
                if (distToDesired < 2.5) {
                    if (!p.arrivedAt) p.arrivedAt = performance.now();
                } else {
                    p.arrivedAt = null;
                }

                if (p.arrivedAt && !p.didPulse) {
                    p.didPulse = true;
                    p.fadeAfter = performance.now() + XREF_POINTER_FADE_MS;
                    _pulseXrefTarget(p.targetEl, XREF_HIGHLIGHT_MS);
                }

                if (p.fadeAfter && performance.now() > p.fadeAfter) {
                    p.dead = true;
                    p.el.classList.remove('active');
                    setTimeout(() => {
                        try { p.el.remove(); } catch (_) {}
                        _xrefActiveKeys.delete(p.targetKey);
                    }, 200);
                }
            }
        }
    }

    // Cleanup
    _xrefPointers = _xrefPointers.filter(p => !p.dead);
    if (_xrefPointers.length > 0) _scheduleXrefTick();
}

function _scheduleXrefTick() {
    if (_xrefTicking) return;
    _xrefTicking = true;
    requestAnimationFrame(_tickXrefPointers);
}

function _spawnXrefPointer({ sourceEl, targetKey, label, openTabName, openSubTabName }) {
    if (!_xrefLayerEl || !_xrefTemplateEl) return;
    if (!sourceEl || !targetKey) return;

    // One link/key = one floating pointer.
    if (_hasActivePointerForKey(targetKey)) return;

    const node = _xrefTemplateEl.content.firstElementChild.cloneNode(true);
    const labelEl = node.querySelector('.xref-pointer-label');

    labelEl.textContent = label || targetKey;

    // Make it clickable: clicking progresses from waiting -> open -> travel.
    node.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();

        // If user clicks the pointer while it's waiting at tab/subtab,
        // we perform the actual tab/subtab open and then start stage 2.
        if (openTabName) openTab(openTabName);
        if (openSubTabName) openSubTab(openSubTabName);
        p.stage = 2;
        _scheduleXrefTick();
    });

    _xrefLayerEl.appendChild(node);

    const sr = sourceEl.getBoundingClientRect();
    const avoidRects = _collectAvoidRectsForPointer(10);
    const start = _pickNonOverlappingPointerPosition(sr, node.offsetWidth || 180, node.offsetHeight || 34, avoidRects);

    const waitEl = _getSubTabButton(openSubTabName) || _getTabButton(openTabName);
    const waitingSelector = waitEl ? (waitEl.id ? `#${waitEl.id}` : null) : null;

    const p = {
        el: node,
        sourceEl,
        targetKey,
        targetEl: null,
        openTabName,
        openSubTabName,
        waitingEl: waitEl || null,
        waitingSelector,
        stage: 1,
        mode: 'waiting',
        dead: false,
        arrivedAt: null,
        didPulse: false,
        fadeAfter: null,
        x: start.px,
        y: start.py,
        desiredX: start.px,
        desiredY: start.px,
    };

    node.style.left = `${p.x}px`;
    node.style.top = `${p.y}px`;
    node.classList.add('active');

    _xrefActiveKeys.add(String(targetKey));
    _xrefPointers.push(p);
    _scheduleXrefTick();
}

function initXrefs() {
    _xrefLayerEl = document.getElementById('xref-layer');
    _xrefTemplateEl = document.getElementById('xref-pointer-template');
    _xrefScrollContainer = document.querySelector('.main-content');

    // Xref link click: spawn a new pointer instance.
    document.addEventListener('click', (e) => {
        const link = e.target.closest('[data-xref]');
        if (!link) return;

        const key = link.getAttribute('data-xref');
        if (!key) return;
        e.preventDefault();

        _spawnXrefPointer({
            sourceEl: link,
            targetKey: key,
            label: link.getAttribute('data-xref-label') || link.textContent?.trim() || key,
            openTabName: link.getAttribute('data-xref-tab') || null,
            openSubTabName: link.getAttribute('data-xref-subtab') || null,
        });
    });

    if (_xrefScrollContainer) {
        _xrefScrollContainer.addEventListener('scroll', _scheduleXrefTick, { passive: true });
    }
    window.addEventListener('resize', _scheduleXrefTick);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            for (const p of _xrefPointers) {
                p.dead = true;
                p.el.classList.remove('active');
                setTimeout(() => p.el.remove(), 150);
            }
            _xrefPointers = [];
            _xrefActiveKeys.clear();
        }
    });
}

// --- PRESETS ---
async function fetchPresets() {
    try {
        const res = await fetch(`${API_BASE}/presets`);
        const presets = await res.json();
        const select = document.getElementById("preset_selector");
        
        // Keep default option
        select.innerHTML = '<option value="">-- Select a Preset --</option>';
        
        presets.forEach(p => {
            const opt = document.createElement("option");
            opt.value = p;
            opt.textContent = p;
            select.appendChild(opt);
        });

        refreshCustomDropdown(select);
    } catch (e) {
        console.error("Failed to fetch presets", e);
    }
}

async function loadSelectedPreset() {
    const select = document.getElementById("preset_selector");
    const presetName = select.value;
    if (!presetName) {
        showNotification("Please select a preset to load.", 'error');
        return;
    }
    
    try {
        const res = await fetch(`${API_BASE}/presets/${presetName}`);
        const data = await res.json();
        _latestConfigSnapshot = data || {};
        applyConfigToForm(data);

        // Trigger updates
        updateNetworkFields();
        buildDynamicLossFields();
        buildDynamicOptimizationFields();
        renderOptimizerArgHints();
        syncCustomDropdownsWithin(document.getElementById("training-form"));
        showNotification(`Loaded preset: ${presetName}`, 'success');
    } catch (e) {
        console.error("Failed to load preset", e);
        showNotification("Failed to load preset", 'error');
    }
}

function applyConfigToForm(config) {
    const form = document.getElementById("training-form");
    if (!form || !config) return;

    const parseBool = (v) => {
        if (v === true || v === false) return v;
        if (v === 1 || v === 0) return Boolean(v);
        const s = v === null || v === undefined ? '' : String(v).trim().toLowerCase();
        if (s === 'true' || s === '1' || s === 'yes' || s === 'y' || s === 'on') return true;
        if (s === 'false' || s === '0' || s === 'no' || s === 'n' || s === 'off' || s === '') return false;
        // Fallback: any other non-empty string counts as true
        return Boolean(s);
    };

    const pickSelectValue = (el, value) => {
        const str = value === null || value === undefined ? '' : String(value);
        if (!str) {
            el.value = '';
            el.dispatchEvent(new Event('change', { bubbles: true }));
            return;
        }
        const opts = Array.from(el.options || []);
        const sTrim = str.trim();
        const sLower = sTrim.toLowerCase();

        // Prefer exact value match
        let idx = opts.findIndex(o => o.value === sTrim);
        // Case-insensitive value match
        if (idx === -1) idx = opts.findIndex(o => String(o.value || '').trim().toLowerCase() === sLower);
        // Fallback: match label
        if (idx === -1) idx = opts.findIndex(o => String(o.textContent || o.text || '').trim().toLowerCase() === sLower);

        if (idx !== -1) {
            el.selectedIndex = idx;
            el.dispatchEvent(new Event('change', { bubbles: true }));
            return;
        }
        // If nothing matches, leave existing selection untouched
    };

    const setValue = (el, value) => {
        if (!el) return;
        if (el.type === "checkbox") {
            el.checked = parseBool(value);
        } else if (el.type === "radio") {
            if (String(el.value) === String(value)) el.checked = true;
            // Some radios represent booleans with "true"/"false"
            else if (parseBool(el.value) === parseBool(value)) el.checked = true;
        } else if (el.type === "number") {
            el.value = value !== null && value !== undefined ? value : "";
        } else if (el.tagName === "SELECT") {
            pickSelectValue(el, value);
        } else if (el.tagName === "TEXTAREA") {
            el.value = value !== undefined && value !== null ? value : "";
        } else {
            el.value = value !== undefined && value !== null ? value : "";
        }
    };

    const isElement = (x) => (typeof Element !== 'undefined') && (x instanceof Element);
    const isMultiControl = (x) => {
        if (!x) return false;
        if (isElement(x)) return false;
        if (x instanceof RadioNodeList) return true;
        if (Array.isArray(x)) return true;
        // HTMLFormControlsCollection/NodeList-like
        return typeof x.length === 'number' && typeof x.item === 'function';
    };

    for (const [key, value] of Object.entries(config)) {
        const el = form.elements[key];
        if (!el) continue;

        // Support RadioNodeList / multiple controls with the same name (e.g., duplicated checkboxes)
        if (isMultiControl(el)) {
            Array.from(el).forEach(node => setValue(node, value));
        } else {
            setValue(el, value);
        }
    }

    refreshRangeFills(form);
    
    // Apply special handling for loss optimization auto toggles
    initializeLossAutoToggles(config);
}

function initializeLossAutoToggles(config) {
    // SNR Gamma auto toggle
    const snrGammaAuto = document.getElementById('snr_gamma_auto');
    const snrGammaInput = document.getElementById('snr_gamma_input');
    if (snrGammaAuto && snrGammaInput) {
        const snrValue = config.snr_gamma || config.min_snr_gamma || 0;
        // If value is 5 (the auto default), check auto
        if (snrValue === 5 || snrValue === 5.0) {
            snrGammaAuto.checked = true;
            snrGammaInput.value = 5;
            snrGammaInput.disabled = true;
        } else {
            snrGammaAuto.checked = snrValue > 0;
            snrGammaInput.value = snrValue || 5;
            snrGammaInput.disabled = snrValue === 5;
        }
    }
    
    // Noise offset auto toggle
    const noiseOffsetAuto = document.getElementById('noise_offset_auto');
    const noiseOffsetInput = document.getElementById('noise_offset_input');
    if (noiseOffsetAuto && noiseOffsetInput) {
        const noiseValue = config.noise_offset_strength || 0;
        // Consider "auto" if it's the recommended value
        noiseOffsetAuto.checked = noiseValue > 0 && Math.abs(noiseValue - 0.0357) < 0.001;
        noiseOffsetInput.value = noiseValue;
    }
    
    // Update UI state
    if (typeof updateLossTypeUI === 'function') updateLossTypeUI();
    if (typeof updateAutoZSNRBadge === 'function') updateAutoZSNRBadge();
}

function updateNetworkFields() {
    const type = document.getElementById("network_type").value;
    const lycorisFields = document.getElementById("lycoris_fields");
    
    if (type === "lycoris" || type === "loha" || type === "lokr") {
        lycorisFields.classList.remove("hidden");
    } else {
        lycorisFields.classList.add("hidden");
    }
}

function getFormControl(name) {
    const form = document.getElementById("training-form");
    if (!form) return null;
    const el = form.elements[name];
    if (!el) return null;
    // Handle RadioNodeList / multiple inputs with same name
    if (typeof el.addEventListener === "function") return el;
    if (el instanceof RadioNodeList || Array.isArray(el) || el.length) {
        const first = el[0];
        return first && typeof first.addEventListener === "function" ? first : null;
    }
    return null;
}

function getFormControls(name) {
    const form = document.getElementById("training-form");
    if (!form) return [];
    const el = form.elements[name];
    if (!el) return [];
    if (typeof el.addEventListener === "function") return [el];
    if (el instanceof RadioNodeList || Array.isArray(el) || el.length) {
        return Array.from(el).filter(node => node && typeof node.addEventListener === "function");
    }
    return [];
}

// --- DYNAMIC LOSS / OPTIM FIELDS ---
const LOSS_FIELD_DEFS = [
    {
        name: "loss_type",
        type: "select",
        label: "Loss Type",
        help: "Determines how the error between the generated image and the target is calculated. 'L2' (MSE) is the standard, penalizing large errors heavily for smooth convergence. 'Huber' is a hybrid that is less sensitive to outliers, while 'L1' (MAE) can provide sharper results but may be less stable. L2 is recommended for most training scenarios.",
        options: [
            { value: "l2", label: "L2 (MSE)" },
            { value: "huber", label: "Huber" },
            { value: "smooth_l1", label: "Smooth L1" },
            { value: "l1", label: "L1 (MAE)" },
        ],
    },
    {
        name: "debiased_estimation_loss",
        type: "checkbox",
        label: "Debiased Estimation Loss",
        help: "Corrects for the bias introduced by the Min-SNR weighting strategy, keeping the training mathematically grounded. Note that this can slightly affect convergence speed or the effective learning rate balance.",
    },
    {
        name: "min_snr_gamma",
        type: "number",
        step: "0.1",
        label: "Min SNR Gamma",
        help: "Balances training focus across different noise levels. This is essential for zero-terminal-SNR models like SDXL to learn structural details. A value of 5.0 is recommended for SDXL; setting this to 0 on v-prediction models may result in noisy or broken images.",
        defaultValue: 0,
    },
    {
        name: "snr_gamma",
        type: "number",
        step: "0.1",
        label: "SNR Gamma (Alias)",
        help: "Alternative name for Min SNR Gamma used by some legacy scripts. Same effect as above.",
        defaultValue: 5.0,
    },
    {
        name: "prior_loss_weight",
        type: "number",
        step: "0.1",
        label: "Prior Loss Weight",
        help: "Controls the relative importance of regularization images versus training images. Higher values (e.g., 1.0) strongly preserve the original class (e.g., 'person'), while lower values allow the model to drift more towards the target style. Setting this too low can lead to catastrophic forgetting of the base concept.",
        defaultValue: 1.0,
        showIf: { field: "use_prior_preservation", equals: true },
    },
    {
        name: "scheduled_huber_schedule",
        type: "select",
        label: "Scheduled Huber Schedule",
        help: "Defines how the Huber loss threshold evolves over time, allowing for a transition between robust and precise loss calculations. 'Constant' is the most stable option.",
        options: [
            { value: "constant", label: "Constant" },
            { value: "exponential", label: "Exponential" },
            { value: "snr", label: "SNR-based" },
        ],
    },
    {
        name: "scheduled_huber_c",
        type: "number",
        step: "0.01",
        label: "Scheduled Huber C",
        help: "The threshold where Huber loss transitions from quadratic (L2) to linear (L1), controlling sensitivity to outliers. Lower values increase robustness, while higher values behave more like standard L2 loss.",
        defaultValue: 0.1,
    },
    {
        name: "scheduled_huber_scale",
        type: "number",
        step: "0.1",
        label: "Scheduled Huber Scale",
        help: "Global multiplier for the Huber loss. Used to balance the loss magnitude against other loss components.",
        defaultValue: 1.0,
    },
    {
        name: "scale_weight_norms",
        type: "number",
        step: "0.1",
        label: "Scale Weight Norms",
        help: "Rescales network weights to maintain them within a specific range, which can prevent exploding gradients in deep networks or when using high learning rates. This is typically not required for standard LoRA training.",
    },
];

const OPTIM_FIELD_DEFS = [
    {
        name: "gradient_checkpointing",
        type: "checkbox",
        label: "Gradient Checkpointing",
        help: "Reduces VRAM usage by re-calculating parts of the model during the backward pass instead of storing them in memory. This enables larger batch sizes at the cost of a 20-30% reduction in training speed.",
    },
    {
        name: "max_grad_norm",
        type: "number",
        step: "0.1",
        label: "Max Grad Norm",
        help: "Clips gradient spikes to a specified value to prevent 'exploding gradients,' which can cause the model to produce NaN or black images. A value of 1.0 is the industry standard.",
        defaultValue: 1.0,
    },
    {
        name: "no_half_vae",
        type: "checkbox",
        label: "No Half VAE",
        help: "Forces the VAE to operate in float32 precision to prevent NaN errors (black squares), which are common with the SDXL VAE in lower precision. This improves stability but increases VRAM consumption.",
    },
];

let _latestConfigSnapshot = {};

function _getCurrentValue(def) {
    const form = document.getElementById("training-form");
    if (!form) return def.defaultValue ?? "";

    const existing = form.elements[def.name];
    if (existing) {
        if (existing.type === "checkbox") return existing.checked;
        return existing.value;
    }

    if (_latestConfigSnapshot && Object.prototype.hasOwnProperty.call(_latestConfigSnapshot, def.name)) {
        const val = _latestConfigSnapshot[def.name];
        if (typeof val === "boolean" || typeof val === "number") return val;
        if (val === null || val === undefined) return "";
        return val;
    }

    return def.defaultValue ?? "";
}

function _buildFieldInput(def, value) {
    let input;
    if (def.type === "select") {
        input = document.createElement("select");
        def.options.forEach(opt => {
            const optionEl = document.createElement("option");
            optionEl.value = opt.value;
            optionEl.textContent = opt.label;
            if (String(value) === String(opt.value)) optionEl.selected = true;
            input.appendChild(optionEl);
        });
    } else {
        input = document.createElement("input");
        input.type = def.type;
        if (def.step) input.step = def.step;
        if (value !== undefined && value !== null) {
            if (def.type === "checkbox") input.checked = Boolean(value);
            else input.value = value;
        }
    }

    input.name = def.name;
    if (def.placeholder) input.placeholder = def.placeholder;
    return input;
}

function _applyShowIf(container) {
    const form = document.getElementById("training-form");
    if (!form || !container) return;

    container.querySelectorAll("[data-show-if]").forEach(group => {
        const rule = JSON.parse(group.dataset.showIf || "{}");
        const controller = getFormControl(rule.field);
        let shouldShow = true;
        if (controller) {
            const val = controller.type === "checkbox" ? controller.checked : controller.value;
            if (rule.equals !== undefined) {
                shouldShow = String(val) === String(rule.equals);
            } else if (rule.notEquals !== undefined) {
                shouldShow = String(val) !== String(rule.notEquals);
            }
        }
        group.style.display = shouldShow ? "block" : "none";
    });
}

function buildDynamicFieldSet(containerId, defs) {
    const container = document.getElementById(containerId);
    const form = document.getElementById("training-form");
    if (!container || !form) return;

    const modelType = form.elements["model_type"]?.value || "sdxl";
    container.innerHTML = "";

    defs.forEach(def => {
        if (def.models && !def.models.includes(modelType)) return;

        const value = _getCurrentValue(def);
        const group = document.createElement("div");
        group.className = "form-group";
        if (def.showIf) group.dataset.showIf = JSON.stringify(def.showIf);

        const label = document.createElement("label");
        label.textContent = def.label;

        const input = _buildFieldInput(def, value);
        const help = document.createElement("div");
        help.className = "help-text";
        help.textContent = def.help || "";

        group.appendChild(label);
        group.appendChild(input);
        group.appendChild(help);
        container.appendChild(group);
    });

    initCustomDropdowns(container);
    _applyShowIf(container);

    // Bind visibility updates for dependent fields
    ["use_prior_preservation", "loss_type", "model_type"].forEach(name => {
        const controls = getFormControls(name);
        controls.forEach(el => {
            if (el._dynamicBound) return;
            el.addEventListener("change", () => {
                if (name === "model_type") {
                    buildDynamicLossFields();
                    buildDynamicOptimizationFields();
                } else {
                    _applyShowIf(document.getElementById("dynamic-loss-fields"));
                    _applyShowIf(document.getElementById("dynamic-optim-fields"));
                }
            });
            el._dynamicBound = true;
        });
    });
}

function buildDynamicLossFields() {
    buildDynamicFieldSet("dynamic-loss-fields", LOSS_FIELD_DEFS);
}

function buildDynamicOptimizationFields() {
    buildDynamicFieldSet("dynamic-optim-fields", OPTIM_FIELD_DEFS);
}

// Optimizer arg helper pills
const OPTIMIZER_ARG_SUGGESTIONS = {
    AdamW: ["weight_decay=0.01", "betas=(0.9,0.999)", "eps=1e-8"],
    AdamW8bit: ["weight_decay=0.01", "betas=(0.9,0.999)", "eps=1e-8"],
    Lion: ["weight_decay=0.01", "betas=(0.9,0.99)", "eps=1e-8"],
    Lion8bit: ["weight_decay=0.01", "betas=(0.9,0.99)", "eps=1e-8"],
    DAdaptAdam: ["weight_decay=0.01", "decouple=True"],
    Prodigy: ["weight_decay=0.01", "decouple=True"],
    CAME: ["weight_decay=0.01", "betas=(0.9,0.999)", "eps=1e-8"],
    Adafactor: ["relative_step=True", "scale_parameter=True", "warmup_init=True", "weight_decay=0.0"],
    SGD: ["weight_decay=0.0", "momentum=0.9"],
};

function renderOptimizerArgHints() {
    const form = document.getElementById("training-form");
    if (!form) return;
    const optimizerSelect = getFormControl("optimizer_type");
    const argsInput = form.elements["optimizer_args"];
    if (!optimizerSelect || !argsInput) return;

    const parent = argsInput.parentElement;
    if (!parent) return;

    let container = document.getElementById("optimizer-arg-hints");
    if (!container) {
        container = document.createElement("div");
        container.id = "optimizer-arg-hints";
        container.className = "help-text";
        parent.appendChild(container);
    }

    const opt = optimizerSelect.value || "AdamW";
    const hints = OPTIMIZER_ARG_SUGGESTIONS[opt] || [];
    if (hints.length === 0) {
        container.innerHTML = "";
        return;
    }

    const pills = hints.map(h => `<button type="button" class="pill" data-arg="${h}">${h}</button>`).join("");
    container.innerHTML = `<div class="pill-hint-row"><span class="pill-hint-label">Suggested for ${opt}:</span>${pills}</div>`;

    container.querySelectorAll("button.pill").forEach(btn => {
        btn.onclick = () => {
            const val = btn.dataset.arg;
            const argName = val.split("=")[0];
            // Parse existing args properly (handle values with commas like betas=(0.9,0.999))
            const currentRaw = argsInput.value || "";
            const existingArgs = [];
            let buffer = "";
            let parenDepth = 0;
            for (const ch of currentRaw) {
                if (ch === "(") parenDepth++;
                else if (ch === ")") parenDepth--;
                if (ch === "," && parenDepth === 0) {
                    if (buffer.trim()) existingArgs.push(buffer.trim());
                    buffer = "";
                } else {
                    buffer += ch;
                }
            }
            if (buffer.trim()) existingArgs.push(buffer.trim());
            
            // Check if this arg name already exists
            const alreadyHas = existingArgs.some(a => a.split("=")[0] === argName);
            if (!alreadyHas) {
                existingArgs.push(val);
                argsInput.value = existingArgs.join(", ");
            }
        };
    });

    if (!optimizerSelect._hintBound) {
        optimizerSelect.addEventListener("change", renderOptimizerArgHints);
        optimizerSelect._hintBound = true;
    }
}

// --- CHART ---
let lossHistory = [];
const MAX_POINTS = 200; // Increased for better resolution

function drawChart() {
    const canvas = document.getElementById("loss-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    
    // Get actual canvas size from parent to ensure it fills the panel
    const panel = canvas.parentElement;
    const rect = panel.getBoundingClientRect();
    
    // Set internal resolution to match display size
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
    }
    
    const w = canvas.width;
    const h = canvas.height;
    
    ctx.clearRect(0, 0, w, h);
    
    const paddingLeft = 50; // Room for Y-axis labels
    const paddingRight = 10;
    const paddingY = 20;
    const chartW = w - paddingLeft - paddingRight;
    const chartH = h - paddingY * 2;

    // Background grid (vertical lines)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = paddingLeft; i <= w - paddingRight; i += 50) { 
        ctx.moveTo(i, paddingY); 
        ctx.lineTo(i, h - paddingY); 
    }
    ctx.stroke();
    
    if (lossHistory.length < 2) return;
    
    // Find min/max with some padding
    let min = Math.min(...lossHistory);
    let max = Math.max(...lossHistory);
    if (min === max) { min -= 0.1; max += 0.1; }
    
    const range = max - min;
    
    // Draw Y-axis labels and horizontal grid
    ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    const numTicks = 5;
    for (let i = 0; i <= numTicks; i++) {
        const val = min + (range * (i / numTicks));
        const y = h - paddingY - (i / numTicks * chartH);
        ctx.fillText(val.toFixed(3), paddingLeft - 10, y + 3);
        
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.beginPath();
        ctx.moveTo(paddingLeft, y);
        ctx.lineTo(w - paddingRight, y);
        ctx.stroke();
    }

    // Draw line with accent color (#CDFF00)
    ctx.strokeStyle = "#CDFF00";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    
    // Calculate step based on current history length vs MAX_POINTS
    const stepX = chartW / (MAX_POINTS - 1);
    
    lossHistory.forEach((val, i) => {
        const x = paddingLeft + (i * stepX);
        const y = h - paddingY - ((val - min) / range * chartH);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    
    ctx.stroke();
    
    // Fill gradient under the line
    const gradient = ctx.createLinearGradient(0, paddingY, 0, h - paddingY);
    gradient.addColorStop(0, "rgba(205, 255, 0, 0.15)");
    gradient.addColorStop(1, "rgba(205, 255, 0, 0)");
    
    ctx.lineTo(paddingLeft + (lossHistory.length - 1) * stepX, h - paddingY);
    ctx.lineTo(paddingLeft, h - paddingY);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
    
    // Glow effect on line
    ctx.shadowColor = "#CDFF00";
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;
    
    // Update overlay
    const lastVal = lossHistory[lossHistory.length - 1];
    const currentLossEl = document.getElementById("current-loss");
    if (currentLossEl) currentLossEl.textContent = lastVal.toFixed(4);
    
    // Also update the big display
    updateLossDisplay(lastVal);
}

function addLossPoint(loss) {
    lossHistory.push(loss);
    if (lossHistory.length > MAX_POINTS) lossHistory.shift();
    drawChart();
}

// --- STATUS & LOGS ---
async function fetchStatus() {
    try {
        const res = await fetch(`${API_BASE}/status`);
        const data = await res.json();
        updateStatusUI(data);
    } catch (e) {
        console.error("Failed to fetch status", e);
    }
}

function updateStatusUI(data) {
    const statusEl = document.getElementById("system-status");
    const systemDot = document.getElementById("system-dot");
    const trainingStatusEl = document.getElementById("training-status");
    const trainingDot = document.getElementById("training-dot");
    const consoleStatusEl = document.getElementById("console-status");
    
    // System Stats in Console
    const vramEl = document.getElementById("console-vram");
    const gpuLoadEl = document.getElementById("console-gpu-load");
    const cpuLoadEl = document.getElementById("console-cpu-load");
    const ramEl = document.getElementById("console-ram");

    if (data.system) {
        if (data.system.gpu && data.system.gpu.has_cuda && data.system.gpu.gpus.length > 0) {
            const gpu = data.system.gpu.gpus[0];
            statusEl.textContent = `${gpu.name} (${gpu.total_memory.toFixed(1)}GB)`;
            
            if (vramEl) {
                const used = gpu.total_memory - gpu.free_memory;
                vramEl.textContent = `${used.toFixed(1)} / ${gpu.total_memory.toFixed(1)} GB`;
            }
            if (gpuLoadEl) {
                // We don't have load in the current hardware.py, but we can show memory usage %
                const usage = ((gpu.total_memory - gpu.free_memory) / gpu.total_memory * 100).toFixed(0);
                gpuLoadEl.textContent = `${usage}% (Mem)`;
            }
        } else {
            statusEl.textContent = data.status === "running" ? "Online" : data.status;
        }

        if (data.system.cpu) {
            if (cpuLoadEl) cpuLoadEl.textContent = `${data.system.cpu.cpu_count} Cores`;
            if (ramEl) {
                const usedRam = data.system.cpu.total_ram - data.system.cpu.available_ram;
                ramEl.textContent = `${usedRam.toFixed(1)} / ${data.system.cpu.total_ram.toFixed(1)} GB`;
            }
        }
    }
    
    systemDot.className = "status-dot " + (data.status === "running" ? "success" : "");
    
    if (data.training) {
        const rawStatus = String(data.training.status || 'idle');
        const prettyStatus = (rawStatus === 'caching_latents')
            ? 'Caching Latents'
            : (rawStatus.charAt(0).toUpperCase() + rawStatus.slice(1));
        trainingStatusEl.textContent = prettyStatus;
        
        // Update console status
        if (consoleStatusEl) {
            consoleStatusEl.textContent = prettyStatus;
        }
        
        const btnToggle = document.getElementById("btn-toggle-train");
        
        if (data.training.is_training) {
            btnToggle.disabled = false;
            btnToggle.textContent = "Stop Training";
            btnToggle.className = "btn-danger";
            trainingDot.className = "status-dot running";
        } else {
            btnToggle.disabled = false;
            btnToggle.textContent = "Start Training";
            btnToggle.className = "btn-primary";
            trainingDot.className = "status-dot" + (data.training.status === "completed" ? " success" : "");
        }

        // Structured training telemetry (NO log parsing)
        updateTrainingTelemetry(data.training);

        // Console output comes from status endpoint
        if (Array.isArray(data.training.logs)) {
            const consoleLog = document.getElementById("console-log");
            if (consoleLog) {
                updateConsoleOutput(data.training.logs.join("\n"));
            }
        }
    }
}

let _lastTelemetryStep = -1;
let _trainingStartTime = null;
let _stepTimes = [];
let _lastStepTimestamp = Date.now();
let _currentRemainingMs = 0;
let _etaInterval = null;
let _telemetryPhase = null;
let _cacheStepTimes = [];
let _lastCacheStep = -1;
let _lastCacheStepTimestamp = 0;

function startETACountdown() {
    if (_etaInterval) return;
    _etaInterval = setInterval(() => {
        const etaEl = document.getElementById("console-eta");
        if (!etaEl || _currentRemainingMs <= 0) return;

        _currentRemainingMs -= 1000;
        if (_currentRemainingMs < 0) _currentRemainingMs = 0;

        const seconds = Math.floor((_currentRemainingMs / 1000) % 60);
        const minutes = Math.floor((_currentRemainingMs / (1000 * 60)) % 60);
        const hours = Math.floor((_currentRemainingMs / (1000 * 60 * 60)) % 24);
        const days = Math.floor(_currentRemainingMs / (1000 * 60 * 60 * 24));

        let etaStr = "";
        if (days > 0) etaStr += `${days}d `;
        etaStr += `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        etaEl.textContent = etaStr;
    }, 1000);
}

function stopETACountdown() {
    if (_etaInterval) {
        clearInterval(_etaInterval);
        _etaInterval = null;
    }
}

function updateTrainingTelemetry(training) {
    const stepEl = document.getElementById("console-step");
    const lossEl = document.getElementById("console-loss");
    const etaEl = document.getElementById("console-eta");
    const progressFill = document.getElementById("console-progress-fill");
    const progressText = document.getElementById("console-progress-text");
    const epochText = document.getElementById("console-epoch-text");

    const isTraining = training.status === "training";
    const isCachingLatents = training.status === "caching_latents";
    const isEtaPhase = isTraining || isCachingLatents;

    // Phase transitions: reset phase-specific timers
    if (_telemetryPhase !== training.status) {
        _telemetryPhase = training.status;
        _currentRemainingMs = 0;
        if (etaEl) etaEl.textContent = "--:--:--";

        if (isTraining) {
            _trainingStartTime = Date.now();
            _stepTimes = [];
            _lastTelemetryStep = -1;
            _lastStepTimestamp = Date.now();
        }

        if (isCachingLatents) {
            _cacheStepTimes = [];
            _lastCacheStep = -1;
            _lastCacheStepTimestamp = Date.now();
        }

        if (!isEtaPhase) {
            _trainingStartTime = null;
            _stepTimes = [];
            _cacheStepTimes = [];
            _currentRemainingMs = 0;
            stopETACountdown();
        }
    }

    if (isEtaPhase) {
        if (_trainingStartTime === null) _trainingStartTime = Date.now();
        startETACountdown();
    }

    if (stepEl && typeof training.current_step === 'number') {
        stepEl.textContent = `${training.current_step} / ${training.total_steps}`;
    }
    
    if (lossEl) {
        if (isTraining && typeof training.loss === 'number') lossEl.textContent = training.loss.toFixed(4);
        else lossEl.textContent = "--";
    }

    // Progress Bar
    if (progressFill && typeof training.progress === 'number') {
        progressFill.style.width = `${training.progress.toFixed(1)}%`;
        if (progressText) progressText.textContent = `${training.progress.toFixed(1)}%`;
    }
    if (epochText && typeof training.epoch === 'number') {
        epochText.textContent = isCachingLatents ? "Latents Cache" : `Epoch ${training.epoch}`;
    }

    // ETA Calculation (Training)
    if (etaEl && isTraining && training.current_step > 0) {
        if (training.current_step !== _lastTelemetryStep) {
            const now = Date.now();
            if (_lastTelemetryStep !== -1) {
                _stepTimes.push(now - _lastStepTimestamp);
                if (_stepTimes.length > 20) _stepTimes.shift();
            }
            _lastStepTimestamp = now;

            if (_stepTimes.length > 0) {
                const avgStepTime = _stepTimes.reduce((a, b) => a + b, 0) / _stepTimes.length;
                const remainingSteps = training.total_steps - training.current_step;
                _currentRemainingMs = remainingSteps * avgStepTime;
            }
        }
    }

    // ETA Calculation (Caching Latents)
    if (etaEl && isCachingLatents && training.current_step >= 0 && training.total_steps > 0) {
        // Only sync on actual progress change (same as training ETA)
        if (training.current_step !== _lastCacheStep) {
            const now = Date.now();

            // If progress jumps backwards (restart), reset sampling.
            if (_lastCacheStep !== -1 && training.current_step < _lastCacheStep) {
                _cacheStepTimes = [];
                _lastCacheStepTimestamp = now;
                _lastCacheStep = training.current_step;
            } else {
                if (_lastCacheStep !== -1) {
                    const deltaSteps = Math.max(1, training.current_step - _lastCacheStep);
                    const deltaMs = Math.max(1, now - _lastCacheStepTimestamp);
                    const perItemMs = deltaMs / deltaSteps;
                    _cacheStepTimes.push(perItemMs);
                    if (_cacheStepTimes.length > 20) _cacheStepTimes.shift();
                }

                _lastCacheStepTimestamp = now;
                _lastCacheStep = training.current_step;
            }

            if (_cacheStepTimes.length > 0) {
                const avgItemMs = _cacheStepTimes.reduce((a, b) => a + b, 0) / _cacheStepTimes.length;
                const remainingItems = Math.max(0, training.total_steps - training.current_step);
                _currentRemainingMs = remainingItems * avgItemMs;
            }
        }
    }

    if (etaEl && !isEtaPhase) {
        etaEl.textContent = "--:--:--";
    }

    // Update chart from structured loss
    if (isTraining && typeof training.loss === 'number' && typeof training.current_step === 'number') {
        if (training.current_step !== _lastTelemetryStep) {
            _lastTelemetryStep = training.current_step;
            addLossPoint(training.loss);
        }
        updateLossDisplay(training.loss, training.current_step, training.epoch);
    }
}

// Global storage
let allModels = [];

// --- MODELS ---
async function fetchModels() {
    try {
        const res = await fetch(`${API_BASE}/models`);
        const models = await res.json();
        allModels = models;
        
        const select = document.getElementById("base_model_name");
        select.innerHTML = "";
        
        models.forEach(m => {
            const opt = document.createElement("option");
            opt.value = m.path;
            opt.textContent = m.name; // user picks architecture manually
            opt.dataset.type = "unknown";
            select.appendChild(opt);
        });

        refreshCustomDropdown(select);

        // Render model cards
        renderModelCards(models.slice(0, 5));
        
        // No auto-detection; user chooses architecture explicitly
    } catch (e) {
        console.error("Failed to fetch models", e);
    }
}

function renderModelCards(models) {
    const container = document.getElementById("model-cards");
    if (!container) return;
    
    container.innerHTML = models.map(m => {
            const icon = m.type === "sdxl" ? "XL" : m.type === "sd3" ? "SD3" : "SD";
            return `
                <div class="model-card" onclick="selectModel('${m.name}')">
                    <div class="model-icon">?</div>
                    <div class="model-name">${m.name}</div>
                    <div class="model-type">manual select</div>
                </div>
            `;
    }).join("");
}

function selectModel(name) {
    const select = document.getElementById("base_model_name");
    select.value = name;
    select.dispatchEvent(new Event('change'));
}

// Models tab removed (keeping model dropdown in Training tab only)

// --- TRAINING ---
function getTrainingFormData() {
    const form = document.getElementById("training-form");
    const formData = new FormData(form);
    const config = {};
    
    // Fields that should be parsed as numbers
    const numFields = [
        "network_dim", "network_alpha", "resolution", "batch_size", 
        "max_train_epochs", "save_every_n_epochs", "gradient_accumulation_steps", 
        "lr_warmup_steps", "seed", "clip_skip", "learning_rate", 
        "unet_lr", "text_encoder_lr", "min_snr_gamma", "noise_offset_strength",
        "noise_offset_random_strength", "adaptive_noise_scale", "multires_noise_iterations", 
        "multires_noise_discount", "caption_dropout_rate", "caption_dropout_every_n_epochs", 
        "keep_tokens", "save_every_n_steps", "sample_every_n_steps", "sample_every_n_epochs",
        "conv_dim", "conv_alpha", "max_train_steps", "lr_scheduler_num_cycles", 
        "lr_scheduler_power", "lr_warmup_ratio", "prior_loss_weight", "scale_weight_norms", 
        "stop_text_encoder_training_pct", "min_bucket_reso", "max_bucket_reso", 
        "bucket_reso_steps", "max_token_length", "vae_batch_size", "scheduled_huber_c", 
        "scheduled_huber_scale", "dora_weight_decay", "module_dropout", "rank_dropout", 
        "network_dropout", "color_aug_strength", "flip_aug_probability", "random_crop_scale",
        "sample_num_inference_steps", "sample_guidance_scale", "sample_seed", 
        "emphasis_strength", "de_emphasis_strength", "num_class_images", 
        "reg_infer_steps", "reg_guidance_scale", "reg_seed", "ema_decay", "max_grad_norm",
        "dataloader_num_workers", "num_validation_images", "num_new_tokens_per_abstraction",
        "train_text_encoder_ti_frac", "train_text_encoder_frac", "logit_mean", "logit_std",
        "mode_scale", "guidance_scale", "max_sequence_length", "snr_gamma", "checkpoints_total_limit",
        "huber_c", "v_pred_like_loss", "ip_noise_gamma", "min_timestep", "max_timestep"
    ];

    // Fields that should be parsed as lists (comma separated)
    const listFields = ["network_args", "optimizer_args"];

    for (let [key, value] of formData.entries()) {
        if (numFields.includes(key)) {
            // Convert to number or null if empty
            config[key] = value.trim() !== "" ? Number(value) : null;
        } else if (listFields.includes(key)) {
            // Convert to list or null if empty
            if (value.trim() !== "") {
                config[key] = value.split(",").map(s => s.trim()).filter(s => s !== "");
            } else {
                config[key] = null;
            }
        } else {
            // Standard string fields - treat empty as empty string unless it's a specific optional field
            // For now, let's keep strings as strings. Pydantic handles Optional[str] = None if we send null, 
            // but if we send "", it might be okay depending on schema.
            // However, for optional strings like metadata, "" is usually fine.
            config[key] = value;
        }
    }
    
    const checkboxes = form.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(cb => {
        // Skip auto-toggle checkboxes (they're UI-only)
        if (cb.id && cb.id.endsWith('_auto')) return;
        config[cb.name] = cb.checked;
    });
    
    // Handle auto SNR gamma
    const snrGammaAuto = document.getElementById('snr_gamma_auto');
    const snrGammaInput = document.getElementById('snr_gamma_input');
    if (snrGammaAuto && snrGammaInput) {
        config.snr_gamma = snrGammaAuto.checked ? 5.0 : Number(snrGammaInput.value) || 0;
    }
    
    return config;
}

async function loadTrainingConfig() {
    try {
        const res = await fetch(`${API_BASE}/training/config`);
        const config = await res.json();
        _latestConfigSnapshot = config || {};
        applyConfigToForm(config);

        updateNetworkFields();
        buildDynamicLossFields();
        buildDynamicOptimizationFields();
        renderOptimizerArgHints();
        const form = document.getElementById("training-form");
        syncCustomDropdownsWithin(form);
        
        // Update Reg Gen Settings visibility
        toggleRegGenSettings();
    } catch (e) {
        console.error("Failed to load training config", e);
    }
}

async function saveTrainingConfig() {
    const config = getTrainingFormData();
    try {
        const res = await fetch(`${API_BASE}/training/config`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config)
        });
        if (res.ok) {
            showNotification("Configuration saved!", 'success');
        } else {
            showNotification("Failed to save configuration.", 'error');
        }
    } catch (e) {
        showNotification(`Error saving config: ${e}`, 'error');
    }
}

async function autoOptimize() {
    try {
        const res = await fetch(`${API_BASE}/system/suggestions`);
        const suggestions = await res.json();
        
        const form = document.getElementById("training-form");
        for (const [key, value] of Object.entries(suggestions)) {
            const input = form.querySelector(`[name="${key}"]`);
            if (input) {
                if (input.type === "checkbox") {
                    input.checked = !!value;
                } else {
                    input.value = value;
                }
            }
        }
        
        showNotification("Applied hardware optimizations!", 'success');
        updateNetworkFields();
    } catch (e) {
        console.error("Failed to get optimizations", e);
        showNotification("Failed to get hardware optimizations", 'error');
    }
}

async function saveSelectedPreset() {
    const select = document.getElementById("preset_selector");
    const presetName = select.value;
    if (!presetName) {
        showNotification("Please select a preset to save to.", 'error');
        return;
    }
    
    const config = getTrainingFormData();
    try {
        const res = await fetch(`${API_BASE}/presets/${presetName}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config)
        });
        if (res.ok) {
            showNotification(`Preset saved: ${presetName}`, 'success');
        } else {
            showNotification("Failed to save preset.", 'error');
        }
    } catch (e) {
        showNotification(`Error saving preset: ${e}`, 'error');
    }
}

async function saveNewPreset() {
    const presetName = document.getElementById("preset-name-input").value.trim();
    if (!presetName) {
        showNotification("Preset name cannot be empty.", 'error');
        return;
    }
    
    // Validate preset name (alphanumeric, underscore, hyphen only)
    if (!/^[a-zA-Z0-9_-]+$/.test(presetName)) {
        showNotification("Preset name can only contain letters, numbers, underscores, and hyphens.", 'error');
        return;
    }
    
    const config = getTrainingFormData();
    try {
        const res = await fetch(`${API_BASE}/presets/${presetName}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config)
        });
        if (res.ok) {
            showNotification(`Preset created: ${presetName}`, 'success');
            closePresetModal();
            // Refresh presets list and select the new one
            await fetchPresets();
            document.getElementById("preset_selector").value = presetName;
            refreshCustomDropdown(document.getElementById("preset_selector"));
        } else {
            showNotification("Failed to create preset.", 'error');
        }
    } catch (e) {
        showNotification(`Error creating preset: ${e}`, 'error');
    }
}

async function deleteSelectedPreset() {
    const select = document.getElementById("preset_selector");
    const presetName = select.value;
    if (!presetName) {
        showNotification("Please select a preset to delete.", 'error');
        return;
    }
    
    // Show modal instead of confirm
    document.getElementById("delete-preset-name").textContent = presetName;
    document.getElementById("delete-preset-modal").classList.remove("hidden");
}

function closePresetModal() {
    document.getElementById("preset-modal").classList.add("hidden");
    document.getElementById("preset-name-input").value = "";
}

function openPresetModal() {
    document.getElementById("preset-modal").classList.remove("hidden");
    document.getElementById("preset-name-input").value = "";
    document.getElementById("preset-name-input").focus();
}

function closeDeleteModal() {
    document.getElementById("delete-preset-modal").classList.add("hidden");
}

async function confirmDeletePreset() {
    const select = document.getElementById("preset_selector");
    const presetName = select.value;
    
    try {
        const res = await fetch(`${API_BASE}/presets/${presetName}`, {
            method: "DELETE"
        });
        if (res.ok) {
            showNotification(`Preset deleted: ${presetName}`, 'success');
            closeDeleteModal();
            await fetchPresets();
            select.value = "";
            refreshCustomDropdown(select);
        } else {
            showNotification("Failed to delete preset.", 'error');
        }
    } catch (e) {
        showNotification(`Error deleting preset: ${e}`, 'error');
    }
}

async function toggleTraining() {
    const btn = document.getElementById("btn-toggle-train");
    if (btn.textContent === "Start Training") {
        await startTraining();
    } else {
        await stopTraining();
    }
}

async function startTraining() {
    const btn = document.getElementById("btn-toggle-train");
    if (btn) btn.disabled = true;

    const config = getTrainingFormData();

    // Trust user-selected architecture; no auto-detection gate

    try {
        const res = await fetch(`${API_BASE}/training/start`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config)
        });
        const result = await res.json();
        if (res.ok) {
            showNotification(`Training started! Job ID: ${result.job_id}`, 'success');
            fetchStatus();
            
            // Update console status
            const statusEl = document.getElementById("console-status");
            if (statusEl) statusEl.textContent = "Starting...";
            
            // Reset console stats
            resetConsoleStats();
        } else {
            showNotification(`Error: ${result.detail}`, 'error');
            if (btn) btn.disabled = false;
        }
    } catch (e) {
        showNotification(`Error starting training: ${e}`, 'error');
        if (btn) btn.disabled = false;
    }
}

async function stopTraining() {
    // Show confirmation modal instead of browser confirm()
    document.getElementById("stop-training-modal").classList.remove("hidden");
}

function closeStopModal() {
    document.getElementById("stop-training-modal").classList.add("hidden");
}

async function confirmStopTraining() {
    const btn = document.getElementById("btn-toggle-train");
    if (btn) btn.disabled = true;
    
    closeStopModal();
    await fetch(`${API_BASE}/training/stop`, { method: "POST" });
    fetchStatus();
    
    // Update console status
    const statusEl = document.getElementById("console-status");
    if (statusEl) statusEl.textContent = "Stopped";
}

function resetConsoleStats() {
    const stepEl = document.getElementById("console-step");
    const lossEl = document.getElementById("console-loss");
    const epochEl = document.getElementById("console-epoch");
    
    if (stepEl) stepEl.textContent = "0";
    if (lossEl) lossEl.textContent = "0.0000";
    if (epochEl) epochEl.textContent = "0";
    
    window._lastParsedStep = -1;
}

// --- METADATA EDITOR ---
async function fetchOutputs() {
    try {
        const res = await fetch(`${API_BASE}/outputs`);
        const files = await res.json();
        const select = document.getElementById("meta-file-select");
        
        // Keep the first option
        select.innerHTML = '<option value="">-- Select a file --</option>';
        
        files.forEach(f => {
            const opt = document.createElement("option");
            // Handle both string (legacy) and object formats
            const name = typeof f === 'string' ? f : f.name;
            opt.value = name;
            opt.textContent = name;
            select.appendChild(opt);
        });

        refreshCustomDropdown(select);
    } catch (e) {
        console.error("Failed to fetch outputs", e);
    }
}

async function loadMetadata() {
    const filename = document.getElementById("meta-file-select").value;
    if (!filename) return;

    try {
        const res = await fetch(`${API_BASE}/metadata/${filename}`);
        if (!res.ok) throw new Error("Failed to load metadata");
        
        const data = await res.json();
        renderMetadataEditor(data);
        const area = document.getElementById("meta-editor-area");
        if (area) area.classList.remove("hidden");
    } catch (e) {
        showNotification(e.message, 'error');
    }
}

function renderMetadataEditor(metadata) {
    const container = document.getElementById("meta-fields");
    container.innerHTML = "";

    // We'll display common fields as inputs, and others as a JSON dump or key-value pairs
    // For simplicity, let's just list them all as key-value inputs
    
    for (const [key, value] of Object.entries(metadata)) {
        const div = document.createElement("div");
        div.className = "form-group";
        div.style.marginBottom = "5px";
        
        const label = document.createElement("label");
        label.textContent = key;
        label.style.fontSize = "0.8em";
        
        const input = document.createElement("input");
        input.type = "text";
        input.value = value;
        input.dataset.key = key;
        input.className = "meta-input";
        
        div.appendChild(label);
        div.appendChild(input);
        container.appendChild(div);
    }
}

async function saveMetadata() {
    const filename = document.getElementById("meta-file-select").value;
    if (!filename) return;

    const inputs = document.querySelectorAll(".meta-input");
    const metadata = {};
    
    inputs.forEach(input => {
        metadata[input.dataset.key] = input.value;
    });

    try {
        const res = await fetch(`${API_BASE}/metadata/${filename}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ metadata })
        });
        
        if (res.ok) {
            showNotification("Metadata saved successfully!", 'success');
        } else {
            showNotification("Failed to save metadata", 'error');
        }
    } catch (e) {
        showNotification("Error saving metadata: " + e, 'error');
    }
}

// --- CONVERSION TOOL ---

async function refreshConversionFiles() {
    const select = document.getElementById('conv_input_file');
    select.innerHTML = '<option value="">Loading...</option>';
    
    // Get output_dir from the training form (the input has name="output_dir", not id)
    const form = document.getElementById('training-form');
    const outputDirInput = form ? form.elements['output_dir'] : null;
    let query = "";
    let outputDir = "project/outputs"; // default
    
    if (outputDirInput && outputDirInput.value) {
        outputDir = outputDirInput.value;
        query = `?dir=${encodeURIComponent(outputDir)}`;
    }
    
    try {
        const res = await fetch(`${API_BASE}/outputs${query}`);

        if (!res.ok) throw new Error(`API error: ${res.status}`);
        
        const files = await res.json();
        
        select.innerHTML = '<option value="">Select a file from outputs...</option>';
        files.forEach(f => {
            if (f.name.endsWith('.safetensors')) {
                const opt = document.createElement('option');
                opt.value = f.path;
                opt.textContent = f.name;
                select.appendChild(opt);
            }
        });
        
        if (files.length === 0) {
            select.innerHTML += '<option disabled>No .safetensors files found</option>';
        }

        // Critical: our UI uses a custom dropdown renderer. Updating the native <select>
        // does NOT automatically update the custom UI; refresh it explicitly.
        if (typeof refreshCustomDropdown === 'function') {
            refreshCustomDropdown(select);
        }
    } catch (e) {
        showNotification(`Failed to load output files: ${e.message}`, "error");
        select.innerHTML = '<option value="">Error loading files (check console)</option>';

        if (typeof refreshCustomDropdown === 'function') {
            refreshCustomDropdown(select);
        }
    }
}

async function startConversion() {
    const filePath = document.getElementById('conv_input_file').value;
    const modelType = document.getElementById('conv_model_type').value;
    const targetFormat = document.getElementById('conv_target_format').value;
    const outputName = document.getElementById('conv_output_name').value;
    
    if (!filePath) {
        showNotification("Please select an input file", "error");
        return;
    }
    
    const btn = document.querySelector('#tab-conversion .btn-primary');
    const originalText = btn.textContent;
    btn.textContent = "Converting...";
    btn.disabled = true;
    
    document.getElementById('conv-result').classList.add('hidden');
    
    try {
        const res = await fetch(`${API_BASE}/tools/convert`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: filePath,
                model_type: modelType,
                target_format: targetFormat,
                output_name: outputName || null
            })
        });
        
        const data = await res.json();
        
        if (res.ok) {
            showNotification("Conversion successful!", "success");
            document.getElementById('conv-result').classList.remove('hidden');
            document.getElementById('conv-result-path').textContent = `Saved to: ${data.output_path}`;
            // Refresh list so the new file appears
            refreshConversionFiles();
        } else {
            showNotification(`Conversion failed: ${data.detail}`, "error");
        }
    } catch (e) {
        console.error(e);
        showNotification(`Conversion error: ${e.message}`, "error");
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

// --- INIT ---
document.addEventListener("DOMContentLoaded", () => {
    fetchStatus();
    fetchModels();
    fetchPresets();
    setInterval(fetchStatus, 2000);

    document.getElementById("btn-toggle-train").addEventListener("click", toggleTraining);
    
    // Preset buttons
    const savePresetBtn = document.getElementById("btn-save-preset");
    if (savePresetBtn) savePresetBtn.addEventListener("click", saveSelectedPreset);
    
    const newPresetBtn = document.getElementById("btn-new-preset");
    if (newPresetBtn) newPresetBtn.addEventListener("click", openPresetModal);
    
    const deletePresetBtn = document.getElementById("btn-delete-preset");
    if (deletePresetBtn) deletePresetBtn.addEventListener("click", deleteSelectedPreset);
    
    loadTrainingConfig();
    
    // Initialize chart
    drawChart();

    // Initialize custom dropdowns once (dynamic refresh happens after fetches)
    initCustomDropdowns();

    // Initialize in-UI cross references (e.g., "Q-LoRA" links)
    initXrefs();

    // Render dynamic sections early (will re-render after config/preset load)
    buildDynamicLossFields();
    buildDynamicOptimizationFields();
    renderOptimizerArgHints();
    initRangeSliders();
    
    // Initialize sample generation slider value displays
    const stepsSlider = document.querySelector('input[name="sample_num_inference_steps"]');
    if (stepsSlider) {
        const updateStepsValue = () => {
            document.getElementById("sample_steps_value").textContent = stepsSlider.value;
        };
        stepsSlider.addEventListener("input", updateStepsValue);
        updateStepsValue();
    }
    
    const guidanceSlider = document.querySelector('input[name="sample_guidance_scale"]');
    if (guidanceSlider) {
        const updateGuidanceValue = () => {
            document.getElementById("sample_guidance_value").textContent = guidanceSlider.value;
        };
        guidanceSlider.addEventListener("input", updateGuidanceValue);
        updateGuidanceValue();
    }
});

// --- NOTIFICATIONS ---
function showNotification(message, type = 'info') {
    const container = document.getElementById('notification-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `notification-toast ${type}`;
    
    let icon = 'INFO';
    if (type === 'success') icon = 'OK';
    if (type === 'error') icon = 'ERR';
    if (type === 'warning') icon = 'WARN';

    toast.innerHTML = `
        <div style="display:flex; align-items:center; gap:10px;">
            <span>${icon}</span>
            <span>${message}</span>
        </div>
    `;

    container.appendChild(toast);

    // Auto remove
    setTimeout(() => {
        toast.classList.add('hiding');
        toast.addEventListener('animationend', () => {
            toast.remove();
        });
    }, 3000);
}

// --- CUSTOM DROPDOWNS ---
let _customDropdownGlobalBound = false;

// Levenshtein utilities for fuzzy dropdown search
function _levenshteinDistance(a, b) {
    const m = a.length;
    const n = b.length;
    if (m === 0) return n;
    if (n === 0) return m;

    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            const cost = a[i - 1] === b[j - 1] ? 0 : 1;
            dp[i][j] = Math.min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            );
        }
    }
    return dp[m][n];
}

function _levenshteinSimilarity(a, b) {
    if (!a && !b) return 1;
    const dist = _levenshteinDistance(a, b);
    const maxLen = Math.max(a.length, b.length, 1);
    return 1 - dist / maxLen;
}

// Allow partial matches: compare against substrings of the target with query length
function _bestLevenshteinSimilarity(query, target) {
    if (!query) return 1;
    const q = String(query).toLowerCase();
    const t = String(target || '').toLowerCase();
    if (!t) return 0;

    let best = _levenshteinSimilarity(q, t);
    const qlen = q.length;
    for (let i = 0; i <= Math.max(0, t.length - qlen); i++) {
        const slice = t.slice(i, i + qlen);
        const sim = _levenshteinSimilarity(q, slice);
        if (sim > best) {
            best = sim;
            if (best >= 1) break;
        }
    }
    return best;
}

function initCustomDropdowns(root = document) {
    if (!_customDropdownGlobalBound) {
        _customDropdownGlobalBound = true;
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.custom-select')) {
                document.querySelectorAll('.custom-select.open').forEach(s => s.classList.remove('open'));
            }
        });
    }

    const selects = root.querySelectorAll('select');
    selects.forEach(select => {
        // Only build once; dynamic option changes should call refreshCustomDropdown(select)
        if (select._customDropdown) return;
        buildCustomDropdown(select);
    });
}

function buildCustomDropdown(select) {
    select.classList.add('custom-initialized');
    select.style.display = 'none';

    const wrapper = document.createElement('div');
    wrapper.className = 'custom-select-wrapper';

    const customSelect = document.createElement('div');
    customSelect.className = 'custom-select';

    const trigger = document.createElement('div');
    trigger.className = 'custom-select-trigger';
    const triggerSpan = document.createElement('span');
    trigger.appendChild(triggerSpan);

    const options = document.createElement('div');
    options.className = 'custom-options';

    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.className = 'custom-search';
    searchInput.placeholder = 'Search…';

    const rebuildOptions = (term = '') => {
        // Remove previous option items but keep the search input in place to avoid losing focus.
        options.querySelectorAll('.custom-option').forEach(node => node.remove());
        if (!options.contains(searchInput)) {
            options.prepend(searchInput);
        }

        const query = term.trim();
        const THRESHOLD = 0.95;

        Array.from(select.options).forEach((option, idx) => {
            const bestSim = Math.max(
                _bestLevenshteinSimilarity(query, option.textContent || option.text || ''),
                _bestLevenshteinSimilarity(query, option.value || '')
            );
            if (query && bestSim < THRESHOLD) return;

            const isSelected = select.selectedIndex === idx;
            const customOption = document.createElement('div');
            customOption.className = `custom-option ${isSelected ? 'selected' : ''}`;
            customOption.dataset.value = option.value;
            customOption.dataset.index = idx;
            customOption.textContent = option.text;

            customOption.addEventListener('click', (e) => {
                e.stopPropagation();
                // Update the select value by index
                select.selectedIndex = idx;
                // Dispatch change event so listeners (and our sync) fire
                select.dispatchEvent(new Event('change', { bubbles: true }));
                customSelect.classList.remove('open');
            });

            options.appendChild(customOption);
        });
    };

    searchInput.addEventListener('input', (e) => {
        rebuildOptions(e.target.value || '');
    });

    trigger.addEventListener('click', (e) => {
        e.stopPropagation();
        document.querySelectorAll('.custom-select.open').forEach(s => {
            if (s !== customSelect) s.classList.remove('open');
        });
        customSelect.classList.toggle('open');
        if (customSelect.classList.contains('open')) {
            searchInput.value = '';
            rebuildOptions('');
            searchInput.focus();
            searchInput.setSelectionRange(searchInput.value.length, searchInput.value.length);
        }
    });

    customSelect.appendChild(trigger);
    customSelect.appendChild(options);
    wrapper.appendChild(customSelect);

    select.parentNode.insertBefore(wrapper, select);

    select._customDropdown = { wrapper, customSelect, trigger, triggerSpan, options, rebuildOptions };
    rebuildOptions();
    syncCustomDropdownFromSelect(select);

    // Keep custom UI in sync when code sets select.value and dispatches change
    select.addEventListener('change', () => syncCustomDropdownFromSelect(select));
}

function syncCustomDropdownFromSelect(select) {
    if (!select || !select._customDropdown) return;
    const { triggerSpan, options } = select._customDropdown;

    // Update trigger text
    if (select.selectedIndex >= 0) {
        triggerSpan.textContent = select.options[select.selectedIndex].text;
    } else {
        // Fallback if value is set but not in options
        triggerSpan.textContent = select.value || '';
    }

    const selectedIndex = select.selectedIndex;
    options.querySelectorAll('.custom-option').forEach(opt => {
        // Use index for robust comparison
        const optIdx = parseInt(opt.dataset.index);
        opt.classList.toggle('selected', optIdx === selectedIndex);
    });
}

function refreshCustomDropdown(select) {
    if (!select) return;
    // Ensure global handlers exist
    initCustomDropdowns();

    // If not built yet, build now
    if (!select._customDropdown) {
        buildCustomDropdown(select);
        return;
    }

    // Rebuild options and resync trigger
    select._customDropdown.rebuildOptions();
    syncCustomDropdownFromSelect(select);
}

function syncCustomDropdownsWithin(root) {
    if (!root) return;
    root.querySelectorAll('select').forEach(select => syncCustomDropdownFromSelect(select));
}

function updateConsoleOutput(logs) {
    const consoleLog = document.getElementById("console-log");
    if (!consoleLog) return;
    
    // logs can be array or string
    const text = Array.isArray(logs) ? logs.join('\n') : (typeof logs === 'string' ? logs : '');
    
    // Only update if content changed
    if (consoleLog._lastContent === text) return;
    consoleLog._lastContent = text;
    
    const lines = text.split('\n').map(l => l.trimEnd()).filter(l => l.trim() !== '');
    let html = '';
    
    for (const line of lines) {
        let className = 'console-line';
        
        if (line.includes('[INFO]')) {
            className += ' info';
        } else if (line.includes('[SUCCESS]')) {
            className += ' success';
        } else if (line.includes('[ERROR]') || line.includes('Error') || line.includes('Traceback')) {
            className += ' error';
        } else if (line.includes('[WARNING]')) {
            className += ' warning';
        } else if (line.includes('Loss:')) {
            className += ' accent';
        }
        
        html += `<span class="${className}">${escapeHtml(line)}</span>\n`;
    }
    
    consoleLog.innerHTML = html || '<span class="console-line muted">No output yet...</span>';
    consoleLog.scrollTop = consoleLog.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function clearConsole() {
    const consoleLog = document.getElementById("console-log");
    if (consoleLog) {
        const now = new Date();
        const pad2 = (n) => String(n).padStart(2, '0');
        const ts = `${String(now.getFullYear()).slice(-2)}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())} ${pad2(now.getHours())}:${pad2(now.getMinutes())}:${pad2(now.getSeconds())}`;
        consoleLog.innerHTML = `<span class="console-line muted">[${ts}] - Console cleared.</span>`;
        consoleLog._lastContent = '';
    }
}

function updateLossDisplay(loss, step, epoch) {
    // Chart overlay update
    const currentLossEl = document.getElementById("current-loss");
    if (currentLossEl) {
        currentLossEl.textContent = loss.toFixed(4);
    }
}

// --- RANGE SLIDERS ---
function updateRangeFill(slider) {
    if (!slider) return;
    const min = Number(slider.min ?? 0);
    const max = Number(slider.max ?? 100);
    const val = Number(slider.value ?? 0);
    const pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
    const clamped = Math.max(0, Math.min(100, pct));
    slider.style.setProperty('--range-fill', `${clamped}%`);
}

function refreshRangeFills(root = document) {
    const sliders = root.querySelectorAll?.('input[type="range"]') || [];
    sliders.forEach(updateRangeFill);
}

function initRangeSliders() {
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const handler = () => updateRangeFill(slider);
        slider.addEventListener('input', handler);
        slider.addEventListener('change', handler);
        updateRangeFill(slider);
    });
}

function toggleRegGenSettings(checkbox) {
    const settingsBlock = document.getElementById('reg-gen-settings');
    if (settingsBlock) {
        const isChecked = checkbox ? checkbox.checked : document.querySelector('input[name="auto_generate_reg_images"]')?.checked;
        settingsBlock.style.display = isChecked ? 'block' : 'none';
    }
}

// ============================================
// LOSS OPTIMIZATION PRESETS & AUTO TOGGLES
// ============================================

const LOSS_PRESETS = {
    default: {
        loss_type: 'l2',
        huber_c: 0.1,
        snr_gamma: 0,
        snr_gamma_auto: false,
        v_pred_like_loss: 0,
        noise_offset_strength: 0,
        noise_offset_auto: false,
        ip_noise_gamma: 0,
        zero_terminal_snr: false,
        debiased_estimation_loss: false,
        scale_v_pred_loss_like_noise_pred: false,
        masked_loss: false
    },
    balanced: {
        loss_type: 'l2',
        huber_c: 0.1,
        snr_gamma: 5,
        snr_gamma_auto: true,
        v_pred_like_loss: 0,
        noise_offset_strength: 0.0357,
        noise_offset_auto: true,
        ip_noise_gamma: 0,
        zero_terminal_snr: true,
        debiased_estimation_loss: false,
        scale_v_pred_loss_like_noise_pred: false,
        masked_loss: false
    },
    quality: {
        loss_type: 'l2',
        huber_c: 0.1,
        snr_gamma: 5,
        snr_gamma_auto: true,
        v_pred_like_loss: 0.1,
        noise_offset_strength: 0.0357,
        noise_offset_auto: true,
        ip_noise_gamma: 0.05,
        zero_terminal_snr: true,
        debiased_estimation_loss: true,
        scale_v_pred_loss_like_noise_pred: false,
        masked_loss: false
    },
    dark_light: {
        loss_type: 'l2',
        huber_c: 0.1,
        snr_gamma: 5,
        snr_gamma_auto: true,
        v_pred_like_loss: 0,
        noise_offset_strength: 0.1,
        noise_offset_auto: false,
        ip_noise_gamma: 0,
        zero_terminal_snr: true,
        debiased_estimation_loss: false,
        scale_v_pred_loss_like_noise_pred: false,
        masked_loss: false
    }
};

function applyLossPreset(presetName) {
    const preset = LOSS_PRESETS[presetName];
    if (!preset) return;
    
    // Update all preset buttons
    document.querySelectorAll('.preset-buttons .btn-sm').forEach(btn => {
        btn.classList.remove('active');
        if (btn.onclick.toString().includes(`'${presetName}'`)) {
            btn.classList.add('active');
        }
    });
    
    // Apply values
    const lossType = document.querySelector('select[name="loss_type"]');
    if (lossType) lossType.value = preset.loss_type;
    
    const huberC = document.querySelector('input[name="huber_c"]');
    if (huberC) huberC.value = preset.huber_c;
    
    const snrGammaAuto = document.getElementById('snr_gamma_auto');
    const snrGammaInput = document.getElementById('snr_gamma_input');
    if (snrGammaAuto && snrGammaInput) {
        snrGammaAuto.checked = preset.snr_gamma_auto;
        snrGammaInput.value = preset.snr_gamma;
        snrGammaInput.disabled = preset.snr_gamma_auto;
    }
    
    const vPredLike = document.querySelector('input[name="v_pred_like_loss"]');
    if (vPredLike) vPredLike.value = preset.v_pred_like_loss;
    
    const noiseOffsetAuto = document.getElementById('noise_offset_auto');
    const noiseOffsetInput = document.getElementById('noise_offset_input');
    if (noiseOffsetAuto && noiseOffsetInput) {
        noiseOffsetAuto.checked = preset.noise_offset_auto;
        noiseOffsetInput.value = preset.noise_offset_strength;
        // Don't disable - let user adjust even with auto
    }
    
    const ipNoiseGamma = document.querySelector('input[name="ip_noise_gamma"]');
    if (ipNoiseGamma) ipNoiseGamma.value = preset.ip_noise_gamma;
    
    const zeroTerminalSnr = document.getElementById('zero_terminal_snr_checkbox');
    if (zeroTerminalSnr) zeroTerminalSnr.checked = preset.zero_terminal_snr;
    
    const debiased = document.querySelector('input[name="debiased_estimation_loss"]');
    if (debiased) debiased.checked = preset.debiased_estimation_loss;
    
    const scaleVPred = document.querySelector('input[name="scale_v_pred_loss_like_noise_pred"]');
    if (scaleVPred) scaleVPred.checked = preset.scale_v_pred_loss_like_noise_pred;
    
    const maskedLoss = document.querySelector('input[name="masked_loss"]');
    if (maskedLoss) maskedLoss.checked = preset.masked_loss;
    
    // Update UI visibility
    updateLossTypeUI();
    updateAutoZSNRBadge();
}

function toggleAutoSNR() {
    const autoCheckbox = document.getElementById('snr_gamma_auto');
    const input = document.getElementById('snr_gamma_input');
    
    if (autoCheckbox && input) {
        if (autoCheckbox.checked) {
            input.disabled = true;
            input.value = 5; // Auto value
        } else {
            input.disabled = false;
        }
    }
}

function toggleAutoNoiseOffset() {
    const autoCheckbox = document.getElementById('noise_offset_auto');
    const input = document.getElementById('noise_offset_input');
    const zsnrCheckbox = document.getElementById('zero_terminal_snr_checkbox');
    
    if (autoCheckbox && input) {
        if (autoCheckbox.checked) {
            // Set recommended value
            input.value = 0.0357;
            // Auto-enable zero terminal SNR
            if (zsnrCheckbox) zsnrCheckbox.checked = true;
        }
    }
    updateAutoZSNRBadge();
}

function updateAutoZSNRBadge() {
    const noiseOffsetAuto = document.getElementById('noise_offset_auto');
    const badge = document.getElementById('zsnr_auto_badge');
    
    if (badge && noiseOffsetAuto) {
        badge.style.display = noiseOffsetAuto.checked ? 'inline' : 'none';
    }
}

function updateLossTypeUI() {
    const lossType = document.querySelector('select[name="loss_type"]');
    const huberGroup = document.getElementById('huber-c-group');
    
    if (lossType && huberGroup) {
        huberGroup.style.display = lossType.value === 'huber' ? 'block' : 'none';
    }
}

// Initialize loss UI on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize auto toggles
    toggleAutoSNR();
    updateLossTypeUI();
    updateAutoZSNRBadge();
    
    // Add listener to noise offset input to auto-toggle ZSNR
    const noiseOffsetInput = document.getElementById('noise_offset_input');
    if (noiseOffsetInput) {
        noiseOffsetInput.addEventListener('change', function() {
            const zsnrCheckbox = document.getElementById('zero_terminal_snr_checkbox');
            if (zsnrCheckbox && parseFloat(this.value) > 0) {
                zsnrCheckbox.checked = true;
            }
        });
    }
});
