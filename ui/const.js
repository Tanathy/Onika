const API_BASE = "/api";

const XREF_HIGHLIGHT_MS = 5000;
const XREF_POINTER_FADE_MS = 900;

const MAX_POINTS = 200; // Increased for better resolution

const AUG_PREVIEW_PER_PAGE = 25;

const LOSS_FIELD_DEFS = [
    {
        name: "loss_type",
        type: "select",
        label: "ui.training.loss.loss_type.label",
        help: "ui.training.loss.loss_type.help",
        options: [
            { value: "l2", label: "ui.training.loss.loss_type.options.l2" },
            { value: "huber", label: "ui.training.loss.loss_type.options.huber" },
            { value: "smooth_l1", label: "ui.training.loss.loss_type.options.smooth_l1" },
            { value: "l1", label: "ui.training.loss.loss_type.options.l1" },
        ],
    },
    {
        name: "debiased_estimation_loss",
        type: "checkbox",
        label: "ui.training.loss.debiased_estimation_loss.label",
        help: "ui.training.loss.debiased_estimation_loss.help",
    },
    {
        name: "min_snr_gamma",
        type: "number",
        step: "0.1",
        label: "ui.training.loss.min_snr_gamma.label",
        help: "ui.training.loss.min_snr_gamma.help",
        defaultValue: 0,
    },
    {
        name: "snr_gamma",
        type: "number",
        step: "0.1",
        label: "ui.training.loss.snr_gamma.label",
        help: "ui.training.loss.snr_gamma.help",
        defaultValue: 5.0,
    },
    {
        name: "prior_loss_weight",
        type: "number",
        step: "0.1",
        label: "ui.training.loss.prior_loss_weight.label",
        help: "ui.training.loss.prior_loss_weight.help",
        defaultValue: 1.0,
        showIf: { field: "use_prior_preservation", equals: true },
    },
    {
        name: "scheduled_huber_schedule",
        type: "select",
        label: "ui.training.loss.scheduled_huber_schedule.label",
        help: "ui.training.loss.scheduled_huber_schedule.help",
        options: [
            { value: "constant", label: "ui.training.loss.scheduled_huber_schedule.options.constant" },
            { value: "exponential", label: "ui.training.loss.scheduled_huber_schedule.options.exponential" },
            { value: "snr", label: "ui.training.loss.scheduled_huber_schedule.options.snr" },
        ],
    },
    {
        name: "scheduled_huber_c",
        type: "number",
        step: "0.01",
        label: "ui.training.loss.scheduled_huber_c.label",
        help: "ui.training.loss.scheduled_huber_c.help",
        defaultValue: 0.1,
    },
    {
        name: "scheduled_huber_scale",
        type: "number",
        step: "0.1",
        label: "ui.training.loss.scheduled_huber_scale.label",
        help: "ui.training.loss.scheduled_huber_scale.help",
        defaultValue: 1.0,
    },
    {
        name: "scale_weight_norms",
        type: "number",
        step: "0.1",
        label: "ui.training.loss.scale_weight_norms.label",
        help: "ui.training.loss.scale_weight_norms.help",
    },
];

const OPTIM_FIELD_DEFS = [
    {
        name: "gradient_checkpointing",
        type: "checkbox",
        label: "ui.training.optim.gradient_checkpointing.label",
        help: "ui.training.optim.gradient_checkpointing.help",
    },
    {
        name: "max_grad_norm",
        type: "number",
        step: "0.1",
        label: "ui.training.optim.max_grad_norm.label",
        help: "ui.training.optim.max_grad_norm.help",
        defaultValue: 1.0,
    },
    {
        name: "no_half_vae",
        type: "checkbox",
        label: "ui.training.optim.no_half_vae.label",
        help: "ui.training.optim.no_half_vae.help",
    },
];

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
