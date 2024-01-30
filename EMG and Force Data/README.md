# Information about the files in WorkshopData folder:

- Files of force data (.lvm) and EMG (.otb+) from 3 different subjects (S1, S2, S3) can be found in this folder. 
- The force files match the EMG files. 
- There are 2 force MVC files related to two tasks of maximum effort, since force files are in Newtons, you can use the MVC file to normalize the force data. 
- The number in the end of the filename is related to the force level performed. If it is equals to 45, it means that file is of a task performed at 45% of MVC. 
- The matrix used for recording the EMG data is: ELSCH064NM4 - with the inter electrode distance of 4mm.

# Information about the experiment protocol:

Participants were seated on a comfortable chair and had their forearm and wrist immobilized with Velcro straps in the experimental apparatus attached to a table. Shoulders were kept relaxed, and the elbow was flexed at 140°. The hand was supported in a vertical position and the index finger was aligned with the forearm. The thumb was tied in a relaxed position over a cylinder apparatus at the same height as the index finger, and the other fingers were positioned embracing the same cylinder apparatus and immobilized with Velcro straps. The visual feedback was provided on an LCD monitor (17”) placed at a ~60 cm distance in front of the subject at the eyes level and the visual feedback gain (18.5 pixels/%MVC) was kept constant throughout the experiment. The target force was shown as a fixed horizontal red line and the force feedback signal was represented by a moving yellow dot. Abduction and flexion of the index finger commanded the y- and x-axis of the yellow dot, respectively. The subject was instructed to perform only abduction force of the index finger during the experiment avoiding as much as possible flexion force. The metacarpophalangeal joint of the index finger was positioned at three different angles: 0° (neutral), +15° (abducted), and -15° (adducted). Here at github there are just files from the neutral positioning. When the joint angle was changed, the height and the orientation of the force sensor was adjusted following the finger position so that the force applied by the index finger was always perpendicular to a small rectangular piece of polylactide (PLA) perpendicularly attached to the surface of the force sensor. The experiment was divided in three blocks (one for each joint angle) and presented in a random order to the participants. Each block consisted of:

1) an estimation of the maximum voluntary contraction (MVC), taken from 2 repetitions of 5 s each with a 60 s rest interval between repetitions (to avoid fatigue);
2) submaximal isometric contractions. Six submaximal contraction intensities (5%, 15%, 30%, 45%, 60%, and 75% MVC) were randomly presented to the participant in each block of the experiment (a new order was drawn for each joint angle). The isometric contraction tasks had 15 s of duration with 45 s of interval between trials to avoid fatigue. From the 15 s trial, the first 5 s were considered the transition period in which the participant had to reach the target force. In the following 10 s, the participant had to maintain performing the target force in which visual feedback was removed in either the second (5 to 10 s) or the third (10 to 15 s) section of the trial (randomly ordered). 
Trials in which the force trespassed a margin of ±20% of the target level were repeated. A green light on the top of the screen indicated the beginning (light on) and the end (light off) of the trial. 


# Information about the visual feedback on each file (not relevant for decomposition):

1) "Force_S1_05.lvm" and "EMG_S1_05.otb+": force at 5% MVC and the visual feedback was present at the first 5 seconds (from 5s to 10s).
2) "Force_S1_45.lvm" and "EMG_S1_45.otb+": force at 45% MVC and the visual feedback was present at the first 5 seconds (from 5s to 10s). 
3) "Force_S2_60.lvm" and "EMG_S2_60.otb+": force at 60% MVC and the visual feedback was present at the last 5 seconds (from 10s to 15s). 
4) "Force_S3_45.lvm" and "EMG_S3_45.otb+": force at 45% MVC and the visual feedback was present at the first 5 seconds (from 5s to 10s). 
 
