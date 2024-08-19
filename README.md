# EMG Crosstalk and Decomposition Workshop

#### Mateus Augusto Schneider Castilhos<sup>1,3</sup>, Carina Marconi Germer<sup>2</sup>, Ellen Pereira Zambalde<sup>1,3</sup>, Leonardo Abdala Elias<sup>1,3</sup>

<sup>1</sup> Department of Electronics and Biomedical Engineering, School of Electrical and Computer Engineering, University of Campinas, Campinas, SP, Brazil

<sup>2</sup> Department of Biomedical Engineering, Federal University of Pernambuco, Recife, PE, Brazil

<sup>3</sup> Neural Engineering Research Laboratory, Center for Biomedical Engineering, University of Campinas, Campinas, SP, Brazil

### About

These notebooks were developed to teach how Principal Component Analysis (PCA) and Independent Component Analysis (ICA) work and how they can be used to decompose high-density surface electromyogram (HD sEMG) signals using the well-known FastICA algorithm. These notebooks were presented on November 3 and 4, 2022, at the "Workshop: Crosstalk and HD EMG Decomposition" organized by the [Neural Engineering Research Laboratory](http://www.fee.unicamp.br/deb/leoelias/ner-lab?language=en), [University of Campinas](http://www.unicamp.br/unicamp/english) (Brazil).

The presenters:

- [Carina Marconi Germer](https://orcid.org/0000-0002-7323-3767)
- [Mateus Augusto Schneider Castilhos](https://orcid.org/0000-0003-2979-5305)
- [Ellen Pereira Zambalde](https://orcid.org/0000-0002-0877-3062)
- [Leonardo Abdala Elias](https://orcid.org/0000-0003-4488-3063)

### Who can use these notebooks?

Anyone who knows the concepts of:

- Basic programming (preferably Python).
- Linear algebra.
- Calculus.
- Random variables and random processes.
- Basic neuroscience, especially related to the neuromuscular system.

**I don't know, and now?**

Don't worry! You can still use it; it will be a little challenging, but as Angelica Montrose said, “Challenges are an opportunity to test you and rise to the next level.”

### What are the objectives of these notebooks?

These notebooks are divided into:

1. The [**EMG_Crosstalk_Decomposition_Workshop.ipynb**](EMG_Crosstalk_Decomposition_Workshop.ipynb) is a didactic notebook that briefly explains the main concepts used to decompose the HD sEMG signals using the FastICA algorithm. These concepts include Blind Source Separation, Principal Component Analysis (PCA) and Whitening, Independent Component Analysis (ICA), and the FastICA algorithm based on concepts of Information Theory such as negative entropy and mutual information. Additionally, this notebook contains several images that make it easier to understand the techniques behind the scenes and code cells that allow the user to interact and practice.

2. The [**Decomposition_Panel.ipynb**](Decomposition_Panel.ipynb) is a notebook with all needed procedures for decomposing the HD sEMG already implemented and some widgets that improve the user experience with the code. This notebook was developed to train researchers and students to analyze the EMG signals, the motor units' spike trains (the result of the algorithm), and to choose some parameters of the algorithm.

### Run they in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MateusASchneiderCastilhos/EMG_Crosstalk_Decomposition_Workshop/blob/main/EMG_Crosstalk_Decomposition_Workshop.ipynb) Open this notebook by signing in to your Google account to open it on Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MateusASchneiderCastilhos/EMG_Crosstalk_Decomposition_Workshop/blob/main/Decomposition_Panel.ipynb) To open the Decomposition Panel, launch this notebook.

If you find an error or a bug, please contact Mateus Augusto Schneider Castilhos by this [e-mail address](mailto:mateus.aschneider@gmail.com?subject=[EMG%20Workshop%20GitHub]%20Bug/error%20founded%20or%20Suggestions) or open an Issue. We will be glad to help you.

## Funding

Part of this project was funded by [CNPq](http://www.cnpq.br/) (Brazilian National Science Foundation): proc. no. 131390/2021-0.

Part of this study was funded by [FAPESP](https://fapesp.br/) (São Paulo Research Foundation): proc. no. 2019/01508-4.

## References

1. COMON, P. Independent component analysis, A new concept?. *Signal Processing*, v. 36, n. 3, p. 287–314, 1994. DOI: [10.1016/0165-1684(94)90029-9](https://doi.org/10.1016/0165-1684(94)90029-9).

2. COMON, P.; JUTTEN, C. Handbook of Blind Source Separation Independent Component Analysis and Applications. First ed. Oxford: *Elsevier*, 2010.

3. HOLOBAR, A.; MINETTO, M. A.; FARINA, D. Accurate identification of motor unit discharge patterns from high-density surface EMG and validation with a novel signal-based performance metric. *Journal of Neural Engineering*, v. 11, n. 1, p. 016008, 2014. DOI: [10.1109/TSP.2007.896108](https://doi.org/10.1109/TSP.2007.896108).

4. HOLOBAR, A.; ZAZULA, D. On the selection of the cost function for gradient-based decomposition of surface electromyograms. *30th Annual International Conference of the IEEE Engineering in Medicine and Biology Society*, Vancouver, BC, Canada, n. 4, p. 4668–4671, 2008. DOI: [10.1109/iembs.2008.4650254](https://doi.org/10.1109/iembs.2008.4650254).

5. HOLOBAR, A.; MINETTO, M. A.; FARINA, D. Experimental Analysis of Accuracy in the Identification of Motor Unit Spike Trains From High-Density Surface EMG. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, v. 18, n. 3, p. 221–229, 2010. DOI: [10.1109/TNSRE.2010.2041593](https://doi.org/10.1109/TNSRE.2010.2041593).

6. HOLOBAR, A.; ZAZULA, D. Multichannel blind source separation using convolution Kernel compensation. *IEEE Transactions on Signal Processing*, v. 55, n. 9, p. 4487–4496, 2007. DOI: [10.1109/TSP.2007.896108](https://doi.org/10.1109/TSP.2007.896108).

7. HYVÄRINEN, A. New approximations of differential entropy for independent component analysis and projection pursuit. *10th International Conference on Neural Information Processing Systems 1997*, Cambridge, MA, United States, MIT Press, 1997. p. 273–279.

8. HYVÄRINEN, A. Fast and robust fixed-point algorithms for independent component analysis. *IEEE Transactions on Neural Networks*, v. 10, n. 3, p. 626–634, 1999. DOI: [10.1109/72.761722](https://doi.org/10.1109/72.761722).

9. HYVÄRINEN, A.; KARHUNEN, J.; OJA, E. Independent Component Analysis. First ed. New York: *John Wiley & Sons*, 2001.

10. HYVÄRINEN, A.; OJA, E. A Fast Fixed-Point Algorithm for Independent Component Analysis. *Neural Computation*, v. 9, p. 1483–1492, 1997. DOI: [10.1109/ICICIP.2010.5564273](https://doi.org/10.1109/ICICIP.2010.5564273).

11. HYVÄRINEN, A.; OJA, E. Independent component analysis: algorithms and applications. *Neural Networks*, v. 13, p. 411–430, 2000. DOI: [10.7819/rbgn.v19i63.1905](https://doi.org/10.7819/rbgn.v19i63.1905).

12. NEGRO, F.; MUCELI, S.; CASTRONOVO, A. M.; et al. Multi-channel intramuscular and surface EMG decomposition by convolutive blind source separation. *Journal of Neural Engineering*, v. 13, n. 2, p. 026027, 2016. DOI: [10.1088/1741-2560/13/2/026027](https://doi.org/10.1088/1741-2560/13/2/026027).

## Aditional References

13. DEL VECCHIO, A.; HOLOBAR, A.; FALLA, D.; et al. Tutorial: Analysis of motor unit discharge characteristics from high-density surface EMG signals. *Journal of Electromyography and Kinesiology*, vol. 53, p. 102426, 2020. DOI: [10.1016/j.jelekin.2020.102426](https://doi.org/10.1016/j.jelekin.2020.102426).

14. ENOKA, R. M. Physiological validation of the decomposition of surface EMG signals. *Journal of Electromyography and Kinesiology*, vol. 46, p. 70-83, 2019. DOI: [10.1016/j.jelekin.2019.03.010](https://doi.org/10.1016/j.jelekin.2019.03.010).

15. FARINA, D.; HOLOBAR, A. Characterization of Human Motor Units From Surface EMG Decomposition. *Proceedings of the IEEE*, vol. 104, no. 2, p. 353-373, 2016. DOI: [10.1109/JPROC.2015.2498665](https://doi.org/10.1109/JPROC.2015.2498665).

16. HOLOBAR, A.; FARINA, D. Blind source identification from the multichannel surface electromyogram. *Physiological Measurement*, vol. 35, no. 7, p. R143--R165, 2014. DOI: [10.1088/0967-3334/35/7/R143](https://doi.org/10.1088/0967-3334/35/7/R143).

17. HOLOBAR, A.; ZAZULA, D. Gradient Convolution Kernel Compensation Applied to Surface Electromyograms. *Independent Component Analysis and Signal Separation, ICA 2007. Lecture Notes in Computer Science*, vol 4666, p. 617–624, 2007. DOI: [10.1007/978-3-540-74494-8_77](https://doi.org/10.1007/978-3-540-74494-8_77).

18. HUG, F.; AVRILLON, S.; DEL VECCHIO, A.; et al. Analysis of motor unit spike trains estimated from high-density surface electromyography is highly reliable across operators. *Journal of Electromyography and Kinesiology*, v. 58, p. 102548, 2021. DOI: [10.1016/j.jelekin.2021.102548](https://doi.org/10.1016/j.jelekin.2021.102548).

19. MENG, L.; CHEN, Q.; JIANG, X.; et al. Evaluation of decomposition parameters for high-density surface electromyogram using fast independent component analysis algorithm. *Biomedical Signal Processing and Control*, v. 75, p. 103615, 2022. DOI: [10.1016/j.bspc.2022.103615](https://doi.org/10.1016/j.bspc.2022.103615).
