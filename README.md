# EMG Crosstalk and Decomposition Workshop

#### Mateus Augusto Schneider Castilhos<sup>1,3</sup>, Carina Marconi Germer<sup>2</sup>, Ellen Pereira Zambalde<sup>1,3</sup>, Leonardo Abdala Elias<sup>1,3</sup>

<sup>1</sup> Department of Electronics and Biomedical Engineering, School of Electrical and Computer Engineering, University of Campinas, Campinas, SP, Brazil

<sup>2</sup> Department of Biomedical Engineering, Federal University of Pernambuco, Recife, PE, Brazil

<sup>3</sup> Neural Engineering Research Laboratory, Center for Biomedical Engineering, University of Campinas, Campinas, SP, Brazil

These notebook were developed to teach how Principal Component Analysis (PCA) and Independent Component Analysis (ICA) work and how they can be used to decompose high-density surface electromyogram (HD sEMG) signals using the well-known FastICA algorithm. These notebooks were presented on November 3 and 4, 2022, at the "Workshop: Crosstalk and HD EMG Decomposition" organized by the [Neural Engineering Research Laboratory](http://www.fee.unicamp.br/deb/leoelias/ner-lab?language=en), [University of Campinas](http://www.unicamp.br/unicamp/english) (Brazil).

The presenters:

- [Carina Marconi Germer](http://lattes.cnpq.br/8205284377032041)
- [Mateus Augusto Schneider Castilhos](http://lattes.cnpq.br/0955125190662270)
- [Ellen Pereira Zambalde](http://lattes.cnpq.br/0063193464498205)
- [Leonardo Abdala Elias](http://lattes.cnpq.br/5429275286295501)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MateusASchneiderCastilhos/EMG_Crosstalk_Decomposition_Workshop/blob/main/EMG_Crosstalk_Decomposition_Workshop.ipynb) Open this notebook by signing in to your Google account to open it on Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MateusASchneiderCastilhos/EMG_Crosstalk_Decomposition_Workshop/blob/main/Decomposition_Panel.ipynb) To open the Decomposition Panel launch this notebook.

If you find an error or a bug, please contact Mateus Augusto Schneider Castilhos by this [e-mail address](mailto:mateus.aschneider@gmail.com?subject=[EMG%20Workshop%20GitHub]%20Bug/error%20founded%20or%20Suggestions) or open an Issue. We will be glad to help you.

## References

COMON, P. Independent component analysis, A new concept? *Signal Processing*, v. 36, n. 3, p. 287–314, 1994. DOI: [10.1016/0165-1684(94)90029-9](https://doi.org/10.1016/0165-1684(94)90029-9).

COMON, P.; JUTTEN, C. Handbook of Blind Source Separation Independent Component Analysis and Applications. First ed. Oxford: *Elsevier*, 2010.

HOLOBAR, A.; MINETTO, M. A.; FARINA, D. Accurate identification of motor unit discharge patterns from high-density surface EMG and validation with a novel signal-based performance metric. *Journal of Neural Engineering*, v. 11, n. 1, p. 016008, 2014. DOI: [10.1109/TSP.2007.896108](https://doi.org/10.1109/TSP.2007.896108).

HOLOBAR, A.; ZAZULA, D. On the selection of the cost function for gradient-based decomposition of surface electromyograms. *30th Annual International Conference of the IEEE Engineering in Medicine and Biology Society*, Vancouver, BC, Canada, n. 4, p. 4668–4671, 2008. DOI: [10.1109/iembs.2008.4650254](https://doi.org/10.1109/iembs.2008.4650254).

HOLOBAR, A.; MINETTO, M. A.; FARINA, D. Experimental Analysis of Accuracy in the Identification of Motor Unit Spike Trains From High-Density Surface EMG. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, v. 18, n. 3, p. 221–229, 2010. DOI: [10.1109/TNSRE.2010.2041593](https://doi.org/10.1109/TNSRE.2010.2041593).

HOLOBAR, A.; ZAZULA, D. Multichannel blind source separation using convolution Kernel compensation. *IEEE Transactions on Signal Processing*, v. 55, n. 9, p. 4487–4496, 2007. DOI: [10.1109/TSP.2007.896108](https://doi.org/10.1109/TSP.2007.896108).

HYVÄRINEN, A. New approximations of differential entropy for independent component analysis and projection pursuit. *10th International Conference on Neural Information Processing Systems 1997*, Cambridge, MA, United States, MIT Press, 1997. p. 273–279.

HYVÄRINEN, A. Fast and robust fixed-point algorithms for independent component analysis. *IEEE Transactions on Neural Networks*, v. 10, n. 3, p. 626–634, 1999. DOI: [10.1109/72.761722](https://doi.org/10.1109/72.761722).

HYVÄRINEN, A.; KARHUNEN, J.; OJA, E. Independent Component Analysis. First ed. New York: *John Wiley & Sons*, 2001.

HYVÄRINEN, A.; OJA, E. A Fast Fixed-Point Algorithm for Independent Component Analysis. *Neural Computation*, v. 9, p. 1483–1492, 1997. DOI: [10.1109/ICICIP.2010.5564273](https://doi.org/10.1109/ICICIP.2010.5564273).

HYVÄRINEN, A.; OJA, E. Independent component analysis: algorithms and applications. *Neural Networks*, v. 13, p. 411–430, 2000. DOI: [10.7819/rbgn.v19i63.1905](https://doi.org/10.7819/rbgn.v19i63.1905).

NEGRO, F.; MUCELI, S.; CASTRONOVO, A. M.; HOLOBAR, A.; FARINA, D. Multi-channel intramuscular and surface EMG decomposition by convolutive blind source separation. *Journal of Neural Engineering*, v. 13, n. 2, p. 026027, 2016. DOI: [10.1088/1741-2560/13/2/026027](https://doi.org/10.1088/1741-2560/13/2/026027).