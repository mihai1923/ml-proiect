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
