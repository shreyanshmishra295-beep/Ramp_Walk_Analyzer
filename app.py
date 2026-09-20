import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- PAGE SETUP ----------------

st.set_page_config(
    page_title="AI Ramp Walk Analyzer",
    page_icon="👠",
    layout="wide"
)

st.sidebar.title("About Project")

st.sidebar.write(
    "AI Ramp Walk Performance Analyzer "
    "analyzes walking posture and movement "
    "using pose landmarks."
)

st.sidebar.write("Scoring Areas:")
st.sidebar.write("• Posture")
st.sidebar.write("• Shoulder Movement")
st.sidebar.write("• Arm Movement")
st.sidebar.write("• Walking Stability")
st.sidebar.write("• Walking Consistency")
st.sidebar.write("• Head Presentation")

st.title("AI Ramp Walk Performance Analyzer")
st.write(
    "Upload a ramp walk video and get an automatic performance dashboard."
)

# ---------------- VIDEO UPLOAD ----------------

uploaded_video = st.file_uploader(
    "Upload your ramp walk video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:

    st.success("Video uploaded successfully!")

    st.video(uploaded_video)

    if st.button("Analyze Video"):

        st.info("AI analysis started...")
        st.write("Processing video... Please wait.")

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:

            temp_file.write(uploaded_video.getbuffer())
            video_path = temp_file.name

        # Model path
        model_path = os.path.join(
            os.path.dirname(__file__),
            "pose_landmarker_full.task"
        )

        if not os.path.exists(model_path):

            st.error(
                "Pose model file not found!"
            )

        else:

            # ---------------- MEDIAPIPE ----------------

            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=model_path
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO
            )

            landmarker = (
                mp.tasks.vision.PoseLandmarker
                .create_from_options(options)
            )

            cap = cv2.VideoCapture(video_path)

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            # ---------------- DATA ----------------

            posture_values = []
            shoulder_values = []
            arm_values = []
            hip_values = []
            head_values = []

            detected_frames = 0
            frame_count = 0

            # ---------------- FRAME ANALYSIS ----------------

            # ---------------- OUTPUT VIDEO ----------------

            output_video_path = os.path.join(
    tempfile.gettempdir(),
    "ramp_walk_skeleton.avi"
)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps,
                (width, height)
            )

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(
                    frame_count * 1000 / fps
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                if result.pose_landmarks:

                    detected_frames += 1

                    pose = result.pose_landmarks[0]

                    nose = pose[0]

                    left_shoulder = pose[11]
                    right_shoulder = pose[12]

                    left_wrist = pose[15]
                    right_wrist = pose[16]

                    left_hip = pose[23]
                    right_hip = pose[24]

                    # -------- POSTURE --------

                    shoulder_y = (
                        left_shoulder.y +
                        right_shoulder.y
                    ) / 2

                    hip_y = (
                        left_hip.y +
                        right_hip.y
                    ) / 2

                    posture_values.append(
                        abs(shoulder_y - hip_y)
                    )

                    # -------- SHOULDER --------

                    shoulder_center = (
                        left_shoulder.x +
                        right_shoulder.x
                    ) / 2

                    shoulder_values.append(
                        shoulder_center
                    )

                    # -------- ARM --------

                    left_arm = abs(
                        left_wrist.x -
                        left_shoulder.x
                    )

                    right_arm = abs(
                        right_wrist.x -
                        right_shoulder.x
                    )

                    arm_values.append(
                        (left_arm + right_arm) / 2
                    )

                    # -------- WALKING STABILITY --------

                    hip_center = (
                        left_hip.x +
                        right_hip.x
                    ) / 2

                    hip_values.append(
                        hip_center
                    )

                    # -------- HEAD --------

                    shoulder_center_x = (
                        left_shoulder.x +
                        right_shoulder.x
                    ) / 2

                    head_values.append(
                        abs(
                            nose.x -
                            shoulder_center_x
                        )
                    )
                out.write(frame)

                frame_count += 1

            out.release()
            st.write("Output video path:",output_video_path)
            st.write("Output video size:",os.path.getsize(output_video_path))
            cap.release()
            landmarker.close()

            # ---------------- CHECK DETECTION ----------------

            if detected_frames == 0:

                st.error(
                    "No human pose detected in this video."
                )

            else:

                # ---------------- CALCULATE MEASUREMENTS ----------------

                posture_avg = np.mean(
                    posture_values
                )

                shoulder_variation = np.std(
                    shoulder_values
                )

                arm_variation = np.std(
                    arm_values
                )

                stability_variation = np.std(
                    hip_values
                )

                consistency_variation = np.std(
                    hip_values
                )

                head_avg = np.mean(
                    head_values
                )

                # ---------------- SCORES ----------------

                # Posture
                if posture_avg < 0.20:
                    posture_score = 20
                elif posture_avg < 0.35:
                    posture_score = 16
                elif posture_avg < 0.50:
                    posture_score = 12
                else:
                    posture_score = 8

                # Shoulder Movement
                if shoulder_variation < 0.05:
                    shoulder_score = 20
                elif shoulder_variation < 0.10:
                    shoulder_score = 16
                elif shoulder_variation < 0.20:
                    shoulder_score = 12
                else:
                    shoulder_score = 8

                # Arm Movement
                if arm_variation < 0.05:
                    arm_score = 15
                elif arm_variation < 0.10:
                    arm_score = 12
                elif arm_variation < 0.20:
                    arm_score = 9
                else:
                    arm_score = 6

                # Walking Stability
                if stability_variation < 0.02:
                    stability_score = 20
                elif stability_variation < 0.05:
                    stability_score = 16
                elif stability_variation < 0.10:
                    stability_score = 12
                else:
                    stability_score = 8

                # Walking Consistency
                if consistency_variation < 0.02:
                    consistency_score = 15
                elif consistency_variation < 0.05:
                    consistency_score = 12
                elif consistency_variation < 0.10:
                    consistency_score = 9
                else:
                    consistency_score = 6

                # Head Presentation
                if head_avg < 0.15:
                    head_score = 10
                elif head_avg < 0.25:
                    head_score = 8
                elif head_avg < 0.40:
                    head_score = 6
                else:
                    head_score = 4

                # ---------------- FINAL SCORE ----------------

                total_score = (
                    posture_score +
                    shoulder_score +
                    arm_score +
                    stability_score +
                    consistency_score +
                    head_score
                )

                if total_score >= 80:
                    performance = "GOOD"
                elif total_score >= 60:
                    performance = "AVERAGE"
                else:
                    performance = "NEEDS IMPROVEMENT"

                # ---------------- DASHBOARD ----------------

                st.success(
                    "AI Pose Analysis Completed!"
                )

                st.divider()

                st.header(
                    f"Overall Score: {total_score}/100"
                )

                st.subheader(
                    f"Performance: {performance}"
                )

                st.write(
                    f"Pose Detection Rate: "
                    f"{detected_frames / total_frames * 100:.2f}%"
                )

                st.divider()

                # Score columns

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Posture",
                    f"{posture_score}/20"
                )

                col2.metric(
                    "Shoulder Movement",
                    f"{shoulder_score}/20"
                )

                col3.metric(
                    "Arm Movement",
                    f"{arm_score}/15"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Walking Stability",
                    f"{stability_score}/20"
                )

                col2.metric(
                    "Walking Consistency",
                    f"{consistency_score}/15"
                )

                col3.metric(
                    "Head Presentation",
                    f"{head_score}/10"
                )

                st.divider()

                # ---------------- CHART ----------------

                categories = [
                    "Posture",
                    "Shoulder",
                    "Arm",
                    "Stability",
                    "Consistency",
                    "Head"
                ]

                percentages = [
                    posture_score / 20 * 100,
                    shoulder_score / 20 * 100,
                    arm_score / 15 * 100,
                    stability_score / 20 * 100,
                    consistency_score / 15 * 100,
                    head_score / 10 * 100
                ]

                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )

                bars = ax.bar(
                    categories,
                    percentages
                )

                ax.set_ylim(0, 100)

                ax.set_ylabel(
                    "Performance (%)"
                )

                ax.set_title(
                    "Ramp Walk Performance Analysis",
                    pad=20
                )

                for bar, value in zip(
                    bars,
                    percentages
                ):

                    ax.text(
                        bar.get_x() +
                        bar.get_width() / 2,
                        value + 2,
                        f"{value:.0f}%",
                        ha="center"
                    )

                st.pyplot(fig)

                st.divider()

                # ---------------- FEEDBACK ----------------

                st.header("AI Feedback")

                if posture_score >= 16:
                    st.success(
                        "✓ Posture: Good posture maintained."
                    )
                else:
                    st.warning(
                        "• Posture: Try to maintain better body alignment."
                    )

                if shoulder_score >= 16:
                    st.success(
                        "✓ Shoulder Movement: Controlled movement."
                    )
                else:
                    st.warning(
                        "• Shoulder Movement: Try to reduce unnecessary shoulder movement."
                    )

                if arm_score >= 12:
                    st.success(
                        "✓ Arm Movement: Good arm coordination."
                    )
                else:
                    st.warning(
                        "• Arm Movement: Improve arm coordination."
                    )

                if stability_score >= 16:
                    st.success(
                        "✓ Walking Stability: Stable walking pattern."
                    )
                else:
                    st.warning(
                        "• Walking Stability: Try to maintain a more stable walking line."
                    )

                if consistency_score >= 12:
                    st.success(
                        "✓ Walking Consistency: Consistent movement."
                    )
                else:
                    st.warning(
                        "• Walking Consistency: Try to maintain a more uniform walking rhythm."
                    )

                if head_score >= 8:
                    st.success(
                        "✓ Head Presentation: Good head alignment."
                    )
                else:
                    st.warning(
                        "• Head Presentation: Try to maintain better head alignment."
                    )

                st.divider()

                st.info(
                    "Analysis is based on pose landmarks and "
                    "rule-based movement measurements."
                )

        # Remove temporary video
        if os.path.exists(video_path):
            os.remove(video_path)
