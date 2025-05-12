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
3.  **Conversia in tonuri de gri**: Imaginile sunt convertite in tonuri de gri si sunt redimensionate la o dimensiune standard de 256x256 pixeli.
5.  **Conversia in array numpy**: Imaginea preprocesata este convertita intr-un array numpy pentru a face usoare calculele.

### 1.2. Ce Informatii Am Extras din Imagini?

Dupa ce am separat creierul de fundalul negru (cu o metoda numita Otsu), am calculat pentru fiecare imagine urmatoarele caracteristici statistice din pixelii regiunii creierului:

1.  **`mean_intensity`**: Media intensitatii pixelilor (luminozitate generala).
2.  **`std_intensity`**: Deviatia standard a intensitatii (contrast).
3.  **`min_intensity`**: Intensitatea minima (cel mai intunecat pixel).
4.  **`max_intensity`**: Intensitatea maxima (cel mai luminos pixel).
5.  **`skewness`**: Asimetria distributiei intensitatilor.
6.  **`median_intensity`**: Mediana intensitatii.
7.  **`q1_intensity`**: Valoarea sub care se situeaza 25% din intensitati.
8.  **`iqr_intensity`**: Intervalul interquartilic (diferenta dintre al treilea si primul quartil al intensitatii), masoara robustetea variatiei.
9.  **`tumor_type`**: Eticheta categoriei (ex: 'glioma', 'notumor').

### 1.3. De Ce Aceste Informatii?

Am ales aceste caracteristici statistice simple deoarece:

*   **Descriu imaginea:** Ofera informatii despre luminozitatea generala, contrastul si distributia intensitatilor pixelilor din regiunea de interes a creierului. O tumora poate modifica aceste aspecte.
*   **Sunt relativ robuste:** Mediana si IQR sunt mai putin sensibile la valori extreme izolate.
*   **Sunt practice:** Usor de calculat si interpretat, oferind o baza pentru modelele de clasificare.

### 1.4. Cum Am Salvat Informatiile?

Caracteristicile extrase pentru fiecare imagine, impreuna cu label-ul tumorei (`tumor_type`) sunt stocate intr-un dataframe din pandas. Acest dataframe este apoi salvat intr-un fisier CSV (`brain_tumor_features.csv`), care va constitui setul de date tabelar utilizat pentru etapele urmatoare ale proiectului.

### 1.5. Observatii si Procesari Suplimentare

Initial, caracteristicile (precum media, mediana, minimul intensitatii etc.) au fost calculate direct pe baza tuturor pixelilor din imaginile redimensionate. Cu toate acestea, am observat ca fundalul predominant negru al imaginilor RMN afecta in mod semnificativ aceste caracteristici. De exemplu, `min_intensity` era intotdeauna 0, iar media si mediana intensitatilor erau foarte mult trase in jos de numarul mare de pixeli negri, astfel informatia relevanta despre tumora devenind eronata.

Pentru a adresa aceasta problema si a obtine corect valorile caracteristicilor, s-a implementat o metoda de calcul al threshold-ului pixelilor **Otsu** (din biblioteca `scikit-image`). Pentru fiecare imagine, algoritmul Otsu determina un prag optim pentru a separa pixelii din prim-plan de cei din fundal. Ulterior, toate caracteristicile mentionate anterior au fost recalculate luand in considerare doar pixelii identificati ca apartinand prim-planului. Aceasta implementare nu este cea mai optima, dar este rapida, necesita doar o apelare de functie, `threshold_otsu` si ne ajuta sa avem totusi un set de date mai relevant.

## 2. Pregatirea Datelor pentru Antrenament

Dupa extragerea caracteristicilor in `brain_tumor_features.csv`, scriptul `split_data.py` imparte datele in training si test

1.  **Impartire Train/Test**: Setul de date este impartit:
    *   80% pentru antrenarea modelului (`train.csv`).
    *   20% pentru testarea modelului (`test.csv`).
2.  **Amestecare (Shuffle)**: Datele sunt amestecate aleatoriu inainte de impartire pentru a asigura ca seturile de antrenament si test sunt reprezentative.
3.  **Stratificare**: Impartirea se face stratificat pe baza coloanei `tumor_type`. Acest lucru asigura ca fiecare tip de tumora este prezent in proportii similare atat in setul de antrenament, cat si in cel de test, pentru a nu favoriza o anumita tumora.
4.  **Reproductibilitate**: Se foloseste o valoare fixa pentru `random_state` pentru ca impartirea sa fie identica la fiecare rulare a scriptului.

Fisierele rezultate, `train.csv` si `test.csv`, contin datele gata pentru a fi folosite la antrenarea si evaluarea modelului.
