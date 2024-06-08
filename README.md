# Traffic Signboard Detection and Alert System

## INTRODUCTION:

The development of innovative technology to aid drivers is critical in the ever-changing world of transportation and road safety. “Traffic Signboard Detection and Alert System” is a cutting-edge technical solution aimed to improve driving safety, minimize accidents, and improve traffic flow on our roads. This technology is an important answer to the expanding issues that drivers encounter, such as information overload and distractions, which frequently contribute to accidents, especially on highways and busy metropolitan routes. Modern car safety measures rely heavily on signboard detection technology. It uses cutting-edge machine learning and computer vision algorithms to recognize and interpret numerous traffic signs, such as speed limits, warnings, regulatory notifications, and other important information. This enables cars to perceive their environment and respond appropriately,  llowing drivers to make educated decisions quickly. This system’s true brilliance shows through when combined with a driver alert mechanism. It can measure the driver’s conduct and attention level as it recognizes signboards. When a possible hazard or infraction is detected, the system immediately informs the driver via visual or audible signals, ensuring that the driver remains aware and in compliance with traffic standards.

## PROBLEM DEFINITION:

The increasing number of traffic accidents, often resulting from a lack of compliance with road signs and signals, poses a significant safety concern. Drivers frequently fail to notice and adhere to critical signboards, leading to accidents and traffic violations. In an era of information overload, drivers are increasingly distracted by mobile devices, in-car entertainment, and other distractions. This distraction, coupled with driver fatigue, reduces attentiveness to road signs and hazards, further escalating the risk of accidents. Inconsistent adherence to road signs and regulations leads to inefficient traffic flow, traffic jams, and delays. This inefficiency affects not only individual drivers but also contributes to increased fuel consumption and environmental pollution. The rapid development of autonomous vehicles requires systems that can effectively detect and interpret road signs to ensure the safety of passengers and pedestrians. The accuracy and reliability of signboard detection systems are crucial for the successful integration of autonomous vehicles into our transportation networks.

## SCOPE:

The project will include the creation of a complete system for detecting and identifying a broad variety of traffic signs, such as speed limits, warnings, regulatory signs, and regionally specific signals. The system will be intended to handle signboard data in real-time, ensuring that drivers receive timely alerts and information as they travel the road, hence improving safety. The project scope involves the development of a multi-modal alert system that will use both visual and audible signals to notify drivers of identified signboards and possible risks, making it accessible to all vehicles. The system will be modular and built for integration with a variety of vehicle types, including conventional automobiles and self-driving vehicles, increasing safety and efficiency in a variety of transportation contexts. Extensive field testing will be conducted as part of the project to confirm the system’s efficacy, accuracy, and user-friendliness, ensuring that it achieves project objectives and contributes to better road safety and traffic flow.

## SAMPLE VISUALIZATION:

![image](https://github.com/harshd23/Traffic-Signboard-Detection-and-Alert-System/blob/main/MISC/sample%20data%20visualization.png?raw=true)

## RESULTS:

![On-Road Detection](https://github.com/harshd23/Traffic-Signboard-Detection-and-Alert-System/blob/main/MISC/Inference%20on%20images/2.png?raw=true)
<p align="center">On-Road Detection</p>

![Real-time Detection](https://github.com/harshd23/Traffic-Signboard-Detection-and-Alert-System/blob/main/MISC/Inference%20on%20images/1.png?raw=true)
<p align="center">>Real-time Detection</p>

![Sign-board Detection](https://github.com/harshd23/Traffic-Signboard-Detection-and-Alert-System/blob/main/MISC/Inference%20on%20images/5.png?raw=true)
<p align="center">Sign-board Detection</p>

## CONCLUSION:

Addressing the limits of existing systems and bridging research gaps are critical for improving road safety and traffic management in the field of traffic signboard identification. This project emphasizes the need of comprehensive signboard identification, real-time processing, multi-modal warnings, flexibility, and extensive field testing. By concentrating on these characteristics, we may alter how drivers perceive and respond to road signs, lowering accidents and enhancing traffic flow.

---

## STEPS TO CREATE CONDA ENVIRONMENT AND ACTIVATE THE TENSORFLOW OBJECT DETECTION API:
1. `conda create -n <envname> pip python=3.9`

2. `conda activate <envname>`

3. `python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"`

4. From within TensorFlow/models/research/ (INSIDE CONDA ENVIRONMENT)
   `protoc object_detection/protos/*.proto --python_out=.`

5. From within TensorFlow/models/research/ (INSIDE COMMAND PROMPT)
   `for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.`

6. `pip install cython`

7. `pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`

8. From within TensorFlow/models/research/ (INSIDE GIT BASH TERMINAL)
   `cp object_detection/packages/tf2/setup.py .`

9. `python -m pip install .`

10. `pip install tensorflow==2.9`

11. `pip install tensorflow-addons==0.17.1`

12. `pip install pyarrow==10.0.1`

13. From within TensorFlow/models/research/ (INSIDE CONDA ENVIRONMENT)
    `python object_detection/builders/model_builder_tf2_test.py`

14. If you get an error of the protobuf:-
    Follow these steps:
    - Install the latest protobuf version (in my case is 4.21.1):   
        `pip install --upgrade protobuf`
         
    - Copy builder.py from .../Lib/site-packages/google/protobuf/internal to another folder on your computer (let's say 'Documents')

    - Install a protobuf version that is compatible with your project (for me 3.19.6)
        `pip install protobuf==3.19.6`

    - Copy builder.py from (let's say 'Documents') to Lib/site-packages/google/protobuf/internal (You will find the folder path in your conda environment folder)

15. From within TensorFlow/models/research/ (INSIDE CONDA ENVIRONMENT)
    `python object_detection/builders/model_builder_tf2_test.py`