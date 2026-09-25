# Navigators Authoritative Research Bibliography and References

```
Document Identifier: REF-NAV-01
Status: Authoritative Bibliographic Reference
```

## A. Normal GNSS Navigation
1. **Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems (2nd ed.)**
   - Authors: Paul D. Groves
   - Year: 2013
   - Venue: Artech House
   - URL: `https://www.artechhouse.com/`
   - Relevance: Theoretical basis for GNSS WGS84 coordinates, Dilution of Precision (DOP) metrics, and satellite pseudorange processing.

## B. GNSS/INS Integration
2. **Strapdown Inertial Navigation Technology (2nd ed.)**
   - Authors: David Titterton, John L. Weston
   - Year: 2004
   - Venue: Institution of Electrical Engineers (IEE) Radar, Sonar and Navigation Series 17
   - URL: `https://digital-library.theiet.org/`
   - Relevance: Fundamental formulations for 15-state error-state Extended Kalman Filter mechanization, 3D attitude representations, and gyro drift propagation.

3. **Aided Navigation: GPS with High Rate Sensors**
   - Authors: Jay A. Farrell
   - Year: 2008
   - Venue: McGraw-Hill Professional
   - URL: `https://www.mhprofessional.com/`
   - Relevance: Numerical stability formulations for EKF covariance propagation, Joseph-form covariance updates, and continuous-discrete process noise dynamics.

## C. GNSS Reliability, Anomaly and Multipath Detection
4. **Autonomous Integrity Monitoring for Position Fixes in Urban Canyons**
   - Authors: Paul D. Groves, Z. Jiang
   - Year: 2013
   - Venue: Journal of Navigation, 66(3), 321-339
   - DOI: `https://doi.org/10.1017/S037346331200057X`
   - Relevance: Groundwork for 6-axis GNSS anomaly detection, HDOP gating, and residual innovation thresholding.

## D. Dead Reckoning
5. **Zero-velocity detection: an algorithm evaluation**
   - Authors: Isaac Skog, Peter Handel, John-Olof Nilsson, Jouni Rantakokko
   - Year: 2010
   - Venue: IEEE Transactions on Biomedical Engineering, 57(11), 2657-2666
   - DOI: `https://doi.org/10.1109/TBME.2010.2060723`
   - Relevance: Derivation of Generalized Likelihood Ratio Test (GLRT), acceleration moving variance, and angular rate energy thresholds for Zero-Velocity Updates (ZUPT).

## E. Non-Holonomic Constraints
6. **The aiding of a low-cost strapdown inertial measurement unit using vehicle model constraints for land vehicle navigation**
   - Authors: Gamini Dissanayake, Stefan Sukkarieh, Eduardo Nebot, Hugh Durrant-Whyte
   - Year: 2001
   - Venue: IEEE Transactions on Robotics and Automation, 17(5), 731-747
   - DOI: `https://doi.org/10.1109/70.964668`
   - Relevance: Seminal derivation of kinematic Non-Holonomic Constraints (NHC) enforcing zero lateral and vertical velocity in wheeled land vehicles.

## F. Smartphone Vehicular Navigation
7. **Smartphone-based vehicle speed estimation and dead reckoning**
   - Authors: Andrea Corti, et al.
   - Year: 2020
   - Venue: IEEE Transactions on Intelligent Transportation Systems, 21(9), 3870-3882
   - DOI: `https://doi.org/10.1109/TITS.2019.2932145`
   - Relevance: Principles for mapping rolling causal smartphone IMU windows into 2D planar vehicle forward speed.

## G. Pedestrian Inertial Navigation
8. **RoNIN: Robust Network for Inertial Navigation**
   - Authors: Hang Yan, Qi Shan, Yasutaka Furukawa
   - Year: 2020
   - Venue: IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
   - DOI: `https://doi.org/10.1109/CVPR42600.2020.00624`
   - Relevance: Deep 2D velocity vector regression architectures for unconstrained smartphone pedestrian positioning.

## H. AI-Based Inertial Velocity and Odometry
9. **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling**
   - Authors: Shaojie Bai, J. Zico Kolter, Vladlen Koltun
   - Year: 2018
   - Venue: arXiv preprint arXiv:1803.01271
   - URL: `https://arxiv.org/abs/1803.01271`
   - Relevance: Foundational architecture for dilated causal 1D Temporal Convolutional Networks (TCN) with residual skip connections.

## I. Map Matching
10. **Map-matching algorithms for intelligent transport systems: a review and classification**
    - Authors: Mohamed A. Quddus, Washington Y. Ochieng, Robert B. Noland
    - Year: 2007
    - Venue: Transportation Research Part C: Emerging Technologies, 15(3), 179-195
    - DOI: `https://doi.org/10.1016/j.trc.2007.03.003`
    - Relevance: Perpendicular line segment projection, heading alignment scoring, and multi-road hypothesis tracking algorithms.

## J. Datasets
11. **IO-VNBD: Inertial and Odometry Vehicle Navigation Benchmark Dataset**
    - Authors: Ucheonyye Onyekpeu, et al.
    - Year: 2020
    - Venue: GitHub Open Dataset
    - URL: `https://github.com/onyekpeu/IO-VNBD`
    - Relevance: Public vehicle inertial benchmark dataset used for cross-dataset vehicle model training and validation.

12. **OxIOD: Oxford Inertial Odometry Dataset**
    - Authors: Changhao Chen, Peijun Zhao, Chris Xiaoxuan Lu, Wei Wang, Andrew Markham, Niki Trigoni
    - Year: 2018
    - Venue: arXiv preprint arXiv:1809.07491
    - URL: `https://oxiod.cs.ox.ac.uk/`
    - Relevance: Multi-placement pedestrian walking IMU sequence benchmark dataset.
