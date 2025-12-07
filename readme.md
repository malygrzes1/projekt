Indeks Branż 2.0 - System Wczesnego Ostrzegania
Projekt realizowany w ramach hackathonu dla PKO Banku Polskiego. Rozwiązanie to system analityczny służący do oceny kondycji oraz perspektyw rozwoju branż w Polsce w horyzoncie 12-36 miesięcy.

Cel Projektu
Opracowanie dynamicznego "Indeksu Branż", który łączy historyczne dane finansowe (twarde) z analizą wydarzeń rynkowych w czasie rzeczywistym (dane alternatywne). System wspiera analityków ryzyka kredytowego w identyfikacji sektorów zagrożonych oraz perspektywicznych.

Metodologia
Rozwiązanie opiera się na modelu hybrydowym, integrującym trzy warstwy analityczne:

Fundament Ilościowy (Lagging Indicators):

Analiza trendów na podstawie danych GUS (F-01) z lat 2020-2024.

Wykorzystanie wskaźników rentowności obrotu netto oraz dynamiki upadłości.

Projekcja bazowa (matematyczna) przy użyciu ważonego modelu hybrydowego (Regresja Liniowa + Momentum).

Analiza Ryzyka (Leading Indicators):

Przetwarzanie nieustrukturyzowanych danych tekstowych (newsy gospodarcze, komunikaty giełdowe).

Wykorzystanie modelu LLM do wnioskowania przyczynowo-skutkowego.

Dynamiczna korekta prognoz finansowych na podstawie "Context Injection" (mapowanie wrażliwości branży na czynniki kosztowe i popytowe).

Wizualizacja:

Interaktywny dashboard prezentujący zmiany kondycji branż w czasie rzeczywistym.

Symulacja scenariuszowa ("What-if analysis") wpływu zdarzeń makroekonomicznych na marże sektorowe.

Wymagania Techniczne
Python 3.10+

Klucz API Openai.

Instalacja i Uruchomienie
Klonowanie repozytorium:

Bash

Instalacja zależności:

Bash

pip install -r requirements.txt
Konfiguracja zmiennych środowiskowych: Utwórz plik .env i dodaj klucz API:

Plaintext

OPENAI_KEY

```
export OPEN_AI_KEY="..."

Uruchomienie dashboardu:

Bash

streamlit run main.py


Autorzy: 
Tomasz Krawczyk
Michał Bastrzyk

Igor Gibas
