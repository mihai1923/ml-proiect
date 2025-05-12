# Proiect PCLP3 - Partea I: Clasificarea Tumorilor Cerebrale

Nume: Tuca Mihai-Laurentiu
Grupa: 314CA

## 1. Descrierea Setului de Date

Pentru acest proiect, am utilizat datele aflate in directorul **All_Tumor_Labeled** (https://www.kaggle.com/datasets/orvile/brain-tumor-dataset ; am pus training + test intr-un singur director urmand sa fie separate ulterior). Acesta contine imagini ale creierului uman, clasificate in patru categorii distincte:

*   **Glioma**: Imagini reprezentand tumori de tip gliom.
*   **Meningioma**: Imagini reprezentand tumori de tip meningiom.
*   **No Tumor**: Imagini ale creierului sanatos, fara prezenta tumorilor.
*   **Pituitary**: Imagini reprezentand tumori hipofizare (pituitare).

Setul de date este structurat in subdirectoare, fiecare corespunzand uneia dintre aceste patru clase. Numarul total de imagini este de aproximativ 7023, distribuite astfel:
*   Glioma: ~1621 imagini
*   Meningioma: ~1645 imagini
*   No Tumor: ~2000 imagini
*   Pituitary: ~1757 imagini

Tipul problemei abordate este **clasificare multi-clasa**, unde scopul este de a antrena un model capabil sa prezica tipul de tumora (sau absenta unei tumori) pe baza caracteristicilor extrase din imagini.

Pentru a pregati datele pentru antrenarea unui model de machine learning, am efectuat urmatorii pasi de preprocesare si extragere de caracteristici, implementati in scriptul `preprocess.py`:

### 1.1. Incarcarea si Preprocesarea Imaginilor

1.  **Iterarea prin directoare**: Scriptul parcurge fiecare subdirector (clasa de tumora) din `All_Tumor_Labeled`.
2.  **Citirea imaginilor**: Fiecare imagine este citita folosind biblioteca Pillow (PIL).
3.  **Conversia in tonuri de gri**: Imaginile sunt convertite in tonuri de gri si sunt redimensionate la o dimensiune standard de 128x128 pixeli. Acest lucru asigura datelor de intrare.
5.  **Conversia in array numpy**: Imaginea preprocesata este convertita intr-un array numpy pentru a face usoare calculele.

### 1.2. Extragerea Caracteristicilor (Features)

Pentru fiecare imagine preprocesata, s-au extras urmatoarele caracteristici. Acestea, impreuna cu eticheta clasei (`tumor_type`), alcatuiesc coloanele setului de date final:

1.  **`mean_intensity`**: Media valorilor pixelilor. Indica luminozitatea generala a imaginii. (Tip: Numar real)
2.  **`std_intensity`**: Deviatia standard a valorilor pixelilor. Masoara contrastul general al imaginii; o valoare mai mare indica un contrast mai ridicat. (Tip: Numar real)
3.  **`min_intensity`**: Valoarea minima a intensitatii unui pixel din imagine (cel mai intunecat pixel). (Tip: Numar intreg)
4.  **`max_intensity`**: Valoarea maxima a intensitatii unui pixel din imagine (cel mai luminos pixel). (Tip: Numar intreg)
5.  **`intensity_range`**: Diferenta dintre `max_intensity` si `min_intensity`. Ofera o alta masura a contrastului imaginii. (Tip: Numar intreg)
6.  **`median_intensity`**: Valoarea mediana a intensitatii pixelilor. Spre deosebire de medie, mediana este mai putin sensibila la outlieri din imagine. (Tip: Numar real)
7.  **`q1_intensity`**: 25% dintre pixeli au o valoare a intensitatii mai mica sau egala cu aceasta. (Tip: Numar real)
8.  **`q3_intensity`**: 75% dintre pixeli au o valoare a intensitatii mai mica sau egala cu aceasta. (Tip: Numar real)
9.  **`tumor_type`**: Eticheta categoriei de tumora (de ex., 'glioma', 'meningioma', 'notumor', 'pituitary'). (Tip: Sir de caractere)

### 1.3. De ce aceste caracteristici?

Am ales aceste date simple despre pixeli deoarece:

*   **Descriu imaginea:** Indica luminozitatea generala, intunecimea si variatiile de intensitate. O tumora poate modifica aceste aspecte.
*   **Arata contrastul:** Masoara diferentele dintre zonele luminoase si cele intunecate. Tumorile pot avea un contrast distinct.
*   **Sunt robuste:** Anumite valori (mediana, percentile) ignora mai bine pixelii izolati, eronati.
*   **Sunt practice:** Usor de calculat si interpretat, ideale pentru modelul nostru.

### 1.4. Crearea Setului de Date Tabelar

Caracteristicile extrase pentru fiecare imagine, impreuna cu label-ul tumorei (`tumor_type`) sunt stocate intr-un dataframe din pandas. Acest dataframe este apoi salvat intr-un fisier CSV (`brain_tumor_features.csv`), care va constitui setul de date tabelar utilizat pentru etapele urmatoare ale proiectului.
