from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import List, Tuple

import numpy as np

from .config import ExperimentConfig, ModelConfig
from .data import aligned_switch_indices, generate_switching_sequence, make_features
from .model import PredictiveCircuit, StepStats


def _build_demo(seed: int = 0) -> tuple[np.ndarray, np.ndarray, List[int], PredictiveCircuit]:
    cfg = ExperimentConfig(seed=seed, model_gated=ModelConfig(use_gate=True, seed=seed))
    seq, raw_switches, _ = generate_switching_sequence(
        T=cfg.T,
        switch_points=cfg.switch_points,
        noise=cfg.noise,
        seed=cfg.seed,
    )
    X, Y = make_features(seq, k=cfg.k)
    switches = aligned_switch_indices(raw_switches, k=cfg.k)
    model_kwargs = asdict(cfg.model_gated)
    model_kwargs["input_dim"] = cfg.k
    model = PredictiveCircuit(**model_kwargs)
    return X, Y, switches, model


def run_pygame_demo(seed: int = 0) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("pygame is not installed. Run: pip install pygame") from exc

    X, Y, switches, model = _build_demo(seed=seed)
    k = X.shape[1]

    pygame.init()
    screen = pygame.display.set_mode((1280, 820))
    pygame.display.set_caption("Surprise-modulated predictive plasticity")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 22)
    small = pygame.font.SysFont("arial", 16)

    BG = (247, 247, 247)
    BLACK = (20, 20, 20)
    RED = (205, 70, 60)
    BLUE = (65, 110, 215)
    GREEN = (60, 150, 90)
    GOLD = (210, 160, 40)
    GRAY = (145, 145, 145)
    LIGHT = (230, 230, 230)
    PURPLE = (140, 70, 170)

    input_pos = [(150, 145 + i * 82) for i in range(k)]
    exc_pos = (500, 220)
    inh_pos = (500, 520)
    mem_pos = (500, 370)
    out_pos = (885, 250)
    target_pos = (885, 520)

    speed = 1
    paused = False
    train_online = True
    idx = 0
    history_loss: List[float] = []
    history_gate: List[float] = []
    history_ei: List[float] = []
    last: StepStats | None = None

    def reset() -> None:
        nonlocal idx, history_loss, history_gate, history_ei, last, model
        X_local, Y_local, _, model_local = _build_demo(seed=seed)
        assert X_local.shape == X.shape and Y_local.shape == Y.shape
        idx = 0
        history_loss = []
        history_gate = []
        history_ei = []
        last = None
        model = model_local

    def draw_text(text: str, pos: Tuple[int, int], color=BLACK, f=font) -> None:
        screen.blit(f.render(text, True, color), pos)

    def node_color(v: float, kind: str = "generic") -> Tuple[int, int, int]:
        vv = float(np.clip(v, 0.0, 1.0))
        if kind == "exc":
            return (255, int(220 - 110 * vv), int(135 - 45 * vv))
        if kind == "inh":
            return (int(135 - 50 * vv), int(170 - 50 * vv), 255)
        if kind == "mem":
            return (int(220 - 65 * vv), int(210 - 80 * vv), 255)
        return (int(225 - 120 * vv), int(225 - 120 * vv), int(225 - 120 * vv))

    def draw_trace(values: List[float], rect: Tuple[int, int, int, int], color: Tuple[int, int, int], label: str) -> None:
        x0, y0, w, h = rect
        pygame.draw.rect(screen, (252, 252, 252), rect)
        pygame.draw.rect(screen, BLACK, rect, 1)
        draw_text(label, (x0 + 6, y0 + 4), f=small)
        if len(values) < 2:
            return
        tail = values[-180:]
        vmin, vmax = min(tail), max(tail)
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0
        points = []
        for i, v in enumerate(tail):
            xx = x0 + 6 + int((w - 12) * i / max(1, len(tail) - 1))
            yy = y0 + h - 8 - int((h - 20) * (v - vmin) / (vmax - vmin))
            points.append((xx, yy))
        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, 2)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed = min(25, speed + 1)
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed - 1)
                elif event.key == pygame.K_t:
                    train_online = not train_online
                elif event.key == pygame.K_r:
                    reset()

        if not paused:
            for _ in range(speed):
                if idx >= len(X):
                    reset()
                x = X[idx]
                y = Y[idx]
                if train_online:
                    last = model.step(x, y)
                else:
                    out = model.forward(x)
                    p = out["p"]
                    err = y - p
                    abs_err = abs(err)
                    gate, novelty = model.compute_gate(abs_err)
                    loss = -(y * np.log(p + 1e-8) + (1.0 - y) * np.log(1.0 - p + 1e-8))
                    ei_imb = abs(out["exc_drive"] - out["inh_drive"]) / (out["exc_drive"] + out["inh_drive"] + 1e-8)
                    last = StepStats(
                        p=float(p),
                        target=float(y),
                        loss=float(loss),
                        err=float(err),
                        exc_drive=float(out["exc_drive"]),
                        inh_drive=float(out["inh_drive"]),
                        mem_state=float(out["mem"]),
                        alpha=float(model.alpha),
                        beta=float(model.beta),
                        gate=float(gate),
                        novelty=float(novelty),
                        err_fast=float(model.err_fast),
                        err_slow=float(model.err_slow),
                        ei_imbalance=float(ei_imb),
                    )
                    model.y_prev = float(y)
                    model.mem_prev = float(out["mem"])
                    model.err_fast = model.err_fast_decay * model.err_fast + (1.0 - model.err_fast_decay) * abs_err
                    model.err_slow = model.err_slow_decay * model.err_slow + (1.0 - model.err_slow_decay) * abs_err
                history_loss.append(last.loss)
                history_gate.append(last.gate)
                history_ei.append(last.ei_imbalance)
                idx += 1

        screen.fill(BG)

        pygame.draw.rect(screen, LIGHT, (40, 22, 1185, 70))
        for s in switches:
            xline = 40 + int(1185 * s / len(X))
            pygame.draw.line(screen, GRAY, (xline, 22), (xline, 92), 2)
        current_x = 40 + int(1185 * max(0, idx - 1) / len(X))
        pygame.draw.line(screen, RED, (current_x, 18), (current_x, 96), 3)
        draw_text(f"time = {idx}/{len(X)}", (55, 35))
        draw_text(f"SPACE pause | UP/DOWN speed={speed} | T learning={'on' if train_online else 'off'} | R reset", (205, 35), f=small)
        draw_text("Dashed vertical markers = hidden rule switches", (205, 60), color=GRAY, f=small)

        if last is None:
            pygame.display.flip()
            clock.tick(60)
            continue

        x = X[min(idx - 1, len(X) - 1)]
        max_we = max(1e-6, float(np.max(model.w_exc)))
        max_wi = max(1e-6, float(np.max(model.w_inh)))
        max_wm = max(1e-6, float(np.max(np.abs(model.w_mem))))

        for i, pos in enumerate(input_pos):
            width_e = 1 + int(7 * model.w_exc[i] / max_we)
            width_i = 1 + int(7 * model.w_inh[i] / max_wi)
            width_m = 1 + int(6 * abs(model.w_mem[i]) / max_wm)
            pygame.draw.line(screen, RED, pos, exc_pos, width_e)
            pygame.draw.line(screen, BLUE, pos, inh_pos, width_i)
            pygame.draw.line(screen, PURPLE, pos, mem_pos, width_m)
            val = float(x[i])
            pygame.draw.circle(screen, node_color(val), pos, 24)
            pygame.draw.circle(screen, BLACK, pos, 24, 2)
            draw_text(f"x{i+1}={int(round(val))}", (pos[0] - 22, pos[1] - 40), f=small)

        beta_w = 1 + int(8 * model.beta / max(1.0, model.beta))
        mem_w = 1 + int(8 * abs(model.kappa) / max(1.0, abs(model.kappa)))
        pygame.draw.line(screen, BLUE, inh_pos, out_pos, beta_w)
        pygame.draw.line(screen, PURPLE, mem_pos, out_pos, mem_w)
        pygame.draw.line(screen, RED, exc_pos, out_pos, 6)

        exc_norm = last.exc_drive / (last.exc_drive + last.inh_drive + 1e-8)
        inh_norm = 1.0 / (1.0 + np.exp(-3.0 * (last.inh_drive - 0.5)))
        mem_norm = 0.5 * (last.mem_state + 1.0)
        out_norm = last.p
        target_norm = last.target

        pygame.draw.circle(screen, node_color(exc_norm, "exc"), exc_pos, 44)
        pygame.draw.circle(screen, BLACK, exc_pos, 44, 2)
        draw_text("E unit", (exc_pos[0] - 26, exc_pos[1] - 64), f=small)
        draw_text(f"E = {last.exc_drive:.2f}", (exc_pos[0] - 36, exc_pos[1] - 12), f=small)

        pygame.draw.circle(screen, node_color(mem_norm, "mem"), mem_pos, 44)
        pygame.draw.circle(screen, BLACK, mem_pos, 44, 2)
        draw_text("Memory", (mem_pos[0] - 29, mem_pos[1] - 64), f=small)
        draw_text(f"m = {last.mem_state:.2f}", (mem_pos[0] - 34, mem_pos[1] - 12), f=small)

        pygame.draw.circle(screen, node_color(inh_norm, "inh"), inh_pos, 44)
        pygame.draw.circle(screen, BLACK, inh_pos, 44, 2)
        draw_text("I unit", (inh_pos[0] - 24, inh_pos[1] - 64), f=small)
        draw_text(f"I = {last.inh_drive:.2f}", (inh_pos[0] - 34, inh_pos[1] - 12), f=small)

        pygame.draw.circle(screen, node_color(out_norm), out_pos, 50)
        pygame.draw.circle(screen, BLACK, out_pos, 50, 2)
        draw_text("Predictor", (out_pos[0] - 34, out_pos[1] - 72), f=small)
        draw_text(f"p(next=1) = {last.p:.2f}", (out_pos[0] - 58, out_pos[1] - 12), f=small)

        pygame.draw.circle(screen, node_color(target_norm), target_pos, 46)
        pygame.draw.circle(screen, BLACK, target_pos, 46, 2)
        draw_text("Target", (target_pos[0] - 26, target_pos[1] - 68), f=small)
        draw_text(f"y = {int(round(last.target))}", (target_pos[0] - 14, target_pos[1] - 12), f=small)

        pygame.draw.rect(screen, (240, 240, 240), (670, 330, 490, 295), border_radius=12)
        pygame.draw.rect(screen, BLACK, (670, 330, 490, 295), 2, border_radius=12)
        draw_text("State variables", (690, 345))
        draw_text(f"prediction error = {last.err:+.3f}", (690, 380))
        draw_text(f"novelty = {last.novelty:.3f}", (690, 410))
        draw_text(f"gate = {last.gate:.3f}", (690, 440), color=GOLD)
        draw_text(f"E/I imbalance = {last.ei_imbalance:.3f}", (690, 470))
        draw_text(f"fast |err| = {last.err_fast:.3f}", (690, 500), color=RED)
        draw_text(f"slow |err| = {last.err_slow:.3f}", (690, 530), color=BLUE)
        draw_text(f"alpha = {last.alpha:.3f}", (690, 560), color=GREEN)
        draw_text(f"beta = {last.beta:.3f}", (690, 590), color=BLUE)

        draw_trace(history_loss, (60, 635, 520, 95), RED, "recent loss")
        draw_trace(history_gate, (60, 520, 520, 95), GOLD, "recent gate")
        draw_trace(history_ei, (60, 750, 520, 50), GRAY, "recent E/I imbalance")

        draw_text(
            "Interpretation: red edges excite the predictor, blue edges inhibit it, purple edges carry internal memory,",
            (60, 790),
            f=small,
        )
        draw_text(
            "and the gold gate boosts learning when fast error rises above slow error by more than a margin.",
            (60, 808),
            f=small,
        )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main_pygame_cli() -> None:
    parser = argparse.ArgumentParser(description="Interactive pygame visualization of the predictive plasticity toy.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_pygame_demo(seed=args.seed)
