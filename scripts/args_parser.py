"""
Command-line argument parser for person following robot controller.
"""

import argparse


_VALID_VISION_LOG_COMPONENTS = {"none", "all", "vision.main", "vision.exporter"}


def _normalize_log_components(parser: argparse.ArgumentParser, raw_value: str) -> str:
    parts = [part.strip() for part in raw_value.split(',') if part.strip()]
    if not parts:
        parser.error("--log-components requires at least one value")

    invalid = [part for part in parts if part not in _VALID_VISION_LOG_COMPONENTS]
    if invalid:
        parser.error(
            "--log-components only accepts: none, all, vision.main, vision.exporter"
        )

    unique_parts = list(dict.fromkeys(parts))
    if "none" in unique_parts and len(unique_parts) > 1:
        parser.error("--log-components=none cannot be combined with other values")
    if "all" in unique_parts and len(unique_parts) > 1:
        parser.error("--log-components=all cannot be combined with other values")

    return ",".join(unique_parts)


def parse_args():
    """Parse command-line arguments for the person following system."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--trt-engine', type=str, default='models/yolo11n-pose-fp16.trt', help='TensorRT engine path for pose detection')
    parser.add_argument('--osnet-trt-engine', type=str, default='models/osnet_ain_x1_0.trt',
                        help='TensorRT engine path for OSNet-AIN ReID embeddings')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG messages')
    parser.add_argument('--rotate', type=int, default=0, help='Rotate input image (clockwise): 0, 90, 180, or 270 degrees')
    parser.add_argument(
        '--camera-mode',
        type=str,
        default='single',
        choices=['single'],
        help="Camera mode. Single-camera runtime only."
    )
    parser.add_argument('--follow', action='store_true', help='Enable person following mode')
    parser.add_argument(
        '--follow-backend',
        type=str,
        default='pid',
        choices=['pid', 'mppi'],
        help='Follow backend. pid keeps direct robot commands in-process; mppi exports targets for the ROS 2 sidecar.'
    )
    parser.add_argument('--network-interface', type=str, default='eth0', help='Network interface for robot control')
    parser.add_argument('--motion-lock-frames', type=int, default=10,
                        help='Require this many consecutive matched visual detections before movement is allowed')
    parser.add_argument('--no-auto-reacquire', dest='auto_reacquire', action='store_false', default=True,
                        help='Skip automatic main-person re-selection after the tracked ID is lost')
    parser.add_argument('--target-export-host', type=str, default='127.0.0.1',
                        help='UDP target export host for the MPPI sidecar')
    parser.add_argument('--target-export-port', type=int, default=41234,
                        help='UDP target export port for the MPPI sidecar')
    parser.add_argument('--target-export-rate-hz', type=float, default=15.0,
                        help='UDP target export rate limit for the MPPI sidecar')
    parser.add_argument(
        '--log-components',
        type=str,
        default='none',
        help='Comma-separated vision ECS log allowlist: none, all, vision.main, vision.exporter',
    )
    parser.add_argument(
        '--preview-fps',
        type=float,
        default=6.0,
        help='Maximum preview/render refresh rate in Hz while the control loop runs uncapped',
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Disable OpenCV preview windows so rendering cannot block control',
    )
    parser.add_argument(
        '--rotation-debug',
        action='store_true',
        help='Enable the separate rotation debug visualization window',
    )
    parser.add_argument(
        '--preprocess-backend',
        type=str,
        default='gpu',
        choices=['cpu', 'gpu'],
        help='Image preprocessing backend before TensorRT inference',
    )
    parser.add_argument('--camera-offset-x-m', type=float, default=0.0,
                        help='Forward offset from camera optical center to base_link origin')
    parser.add_argument('--camera-offset-y-m', type=float, default=0.0,
                        help='Left offset from camera optical center to base_link origin')
    parser.add_argument('--ecs-log-dir', type=str, default='logs',
                        help='Directory for ECS JSONL analytics logs')
    # DEBUG-TRACE REMOVE-ME: Temporary structured timing traces for stall debugging.
    parser.add_argument('--debug-trace-dir', type=str, default='',
                        help='Directory for temporary debug-trace JSONL logs (empty disables)')
    parser.add_argument('--debug-trace-every-n-frames', type=int, default=1,
                        help='Emit debug-trace timing every N frames (minimum 1)')
    
    # PID controller parameters for X-axis translation (forward/backward) control
    parser.add_argument('--kp', type=float, default=0.9, help='Proportional gain for X-axis translation (forward/backward) PID controller')
    parser.add_argument('--kd', type=float, default=0.3, help='Derivative gain for X-axis translation (forward/backward) PID controller')
    parser.add_argument('--ki', type=float, default=0.0, help='Integral gain for X-axis translation (forward/backward) PID controller')
    parser.add_argument('--trans-x-max', type=float, default=0.6, help='Maximum X-axis translation (forward/backward) command magnitude')
    parser.add_argument('--trans-x-tolerance', type=float, default=0.3, help='Deadband tolerance (meters) around target distance')
    parser.add_argument('--trans-x-antiwindup', type=float, default=0.0, help='Anti-windup back-calculation gain for X-axis translation PID')
    parser.add_argument('--trans-x-alpha', type=float, default=0.4, help='EMA smoothing factor (0-1] for X-axis translation velocity (higher=faster response)')
    
    # PID controller parameters for rotation control (disabled by default)
    parser.add_argument('--rot-kp', type=float, default=0.0, help='Proportional gain for rotation PID controller')
    parser.add_argument('--rot-kd', type=float, default=0.0, help='Derivative gain for rotation PID controller')
    parser.add_argument('--rot-ki', type=float, default=0.0, help='Integral gain for rotation PID controller')
    parser.add_argument('--rot-max', type=float, default=0.0, help='Maximum rotation command magnitude')
    parser.add_argument('--rot-tolerance', type=float, default=3.0, help='Deadband tolerance (degrees) for centering')
    parser.add_argument('--rot-antiwindup', type=float, default=0.0, help='Anti-windup back-calculation gain for rotation PID')
    parser.add_argument('--rot-alpha', type=float, default=0.0, help='EMA smoothing factor (0-1] for rotation velocity (higher=faster response)')

    # Rotation error penalty parameters (bbox-based)
    parser.add_argument('--edge-penalty-k', type=float, default=10.0, help='Exponential decay constant for edge proximity penalty (higher=stronger)')
    parser.add_argument('--size-penalty-k', type=float, default=8.0, help='Exponential decay constant for small-bbox penalty (higher=stronger)')
    parser.add_argument('--large-bbox-thresh', type=float, default=0.5, help='BBox width/frame ratio where edge/size penalties are suppressed')
    
    # Target settings
    parser.add_argument('--target-distance', type=float, default=0.8, help='Target following distance in meters')

    # ReID controls (always enabled in runtime)
    parser.add_argument('--reid-gallery-size', type=int, default=50, help='Maximum number of embeddings stored in ReID gallery')
    parser.add_argument('--reid-update-interval-sec', type=float, default=2.0, help='Interval for stable-track gallery refresh')
    parser.add_argument('--reid-dedupe-cos', type=float, default=0.990, help='Cosine dedupe threshold vs last accepted gallery feature')
    parser.add_argument('--reid-seed-stable-sec', type=float, default=2.0, help='Required stable tracking time before seeding')
    parser.add_argument('--reid-seed-count', type=int, default=5, help='Number of original crops collected for initial gallery seed')
    parser.add_argument('--reid-lgpr-per-image', type=int, default=2, help='Number of LGPR variants generated per seed image')
    parser.add_argument('--reid-match-thresh', type=float, default=0.85, help='Minimum best candidate score to accept a ReID match')
    parser.add_argument('--reid-match-margin', type=float, default=0.20, help='Minimum margin between best and second-best candidate scores')
    parser.add_argument('--reid-nfc-k1', type=int, default=2, help='NFC first-neighbor count')
    parser.add_argument('--reid-nfc-k2', type=int, default=2, help='NFC mutual-neighbor refinement count')
    parser.add_argument('--reid-reacquire-timeout-sec', type=float, default=5.0,
                        help='Maximum unresolved ReID reacquire time before safe shutdown')
    parser.add_argument('--reid-search-pid-sec', type=float, default=2.0,
                        help='Duration to reuse frozen PID errors during ReID reacquire before stopping motion')
    
    args = parser.parse_args()
    args.log_components = _normalize_log_components(parser, args.log_components)
    args.debug_trace_every_n_frames = max(1, int(args.debug_trace_every_n_frames))
    return args
