# DSP-Sound-analysis-and-reproduction
School project aiming towards sound decoding and reproduction in python.

## Intro
=> Geluidssynthetiseren (in python)
1. Karakteristieken van geluid onderzoeken
2. Genereren door algoritme adhv parameters

## Basistechnieken:
1. Frame
    per ~20ms => frame
    elke frame sinussen/frequentie analyseren&benaderen
2. Event
    Parametrisch decodering (zie minerva documentatie)
      - continu
      - sinussen/frequentie/fase
      - ruis + filters
      - transient (amplitudemodulatie ifv de tijd => meixner verloop)
(3. Fysisch)

## Begeleiding & Timing
* 29 maart: een 2e geluid kiezen naar keuze en doormailen (p.devos@ugent.be)
* 19 april: begeleiding
* 3 mei: begeleiding
* 10 mei: begeleiding
* 24 mei: code ingeven met commentaar (dient als verslag) 

## Code progress
Momenteel blijkt alles te werken op vlak van frequencies. Deze worden correct geëxporteerd
en de vermenigvuldiging werkt ook degelijk.
Nu moet nog gezocht worden naar een ADSR soort gelijk techniek om de envelope parameters
te verkleinen.

## Research
* https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/ParametricWaveField.pdf
* http://www.img.lx.it.pt/pcs2007/presentations/JurgenHere_Sound_Images.pdf
* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.95.2415&rep=rep1&type=pdf Vnf pagina 18
* http://www.phy.mtu.edu/~suits/phaseshifts.html
