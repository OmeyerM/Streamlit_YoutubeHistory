#Importowanie potrzebnych bibliotek
import streamlit as st
from collections import Counter
#import calendar
import locale

import json
#import random
import datetime
import matplotlib.pyplot as plt
#import pytz
import pandas as pd
#import unicodedata
#import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import nltk

from streamlit.components.v1 import html

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# Ustawienie lokalizacji na polską
#locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Odczytaj zbiory danych z plików JSON
with open('youtube_records.json', 'r', encoding='utf-8') as file:
    youtube_records = json.load(file)

#WYKRES 1

# Przygotowanie danych
hour_counts = Counter((record['month'], record['hour']) for record in youtube_records)

# Lista nazw miesięcy
month_names = ['styczeń', 'luty', 'marzec', 'kwiecień', 'maj', 'czerwiec', 'lipiec', 'sierpień', 'wrzesień', 'październik', 'listopad', 'grudzień']

# Funkcja do tworzenia i aktualizacji wykresu
def update_plot(filter_month_name):
    filter_month = month_names.index(filter_month_name) + 1
    filtered_hour_counts = {hour: count for (month, hour), count in hour_counts.items() if month == filter_month}
    hours = list(range(24))
    counts = [filtered_hour_counts.get(hour, 0) for hour in hours]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hours, counts, color=(1, 0, 0, 0.8))
    ax.set_xlabel('Godzina')
    ax.set_ylabel('Ilość filmów')
    ax.set_title(f'Ilość obejrzanych filmów na platformie YouTube w poszczególnych godzinach dla miesiąca {filter_month_name}')
    ax.set_xticks(hours)

    st.pyplot(fig)


#WYKRES 2
# Funkcja do tworzenia i aktualizacji wykresu
def update_day_of_week_plot(filter_month_name):
    # Przetwarzanie danych
    day_of_week_counts = Counter()

    for record in youtube_records:
        # Konwertowanie daty i czasu z formatu "YYYY-MM-DD HH:MM:SS"
        datetime_str = record['time']
        datetime_obj = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        
        # Filtrowanie danych tylko dla wybranego miesiąca
        if datetime_obj.month == month_names.index(filter_month_name) + 1:
            # Pobieranie nazwy dnia tygodnia
            day_name = datetime_obj.strftime('%A')
            
            # Zliczanie ilości filmów dla danego dnia tygodnia
            day_of_week_counts[day_name] += 1

    days_of_week = list(day_of_week_counts.keys())
    view_counts = list(day_of_week_counts.values())

    data = pd.DataFrame({'Dzień tygodnia': days_of_week, 'Ilość filmów': view_counts})

    #order = ['poniedziałek', 'wtorek', 'środa', 'czwartek', 'piątek', 'sobota', 'niedziela']
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    data['Dzień tygodnia'] = pd.Categorical(data['Dzień tygodnia'], categories=order, ordered=True)

    data = data.sort_values('Dzień tygodnia')

    filter_month = month_names.index(filter_month_name) + 1


    plt.figure(figsize=(10, 5.55))
    plt.bar(data['Dzień tygodnia'], data['Ilość filmów'], color=(1, 0, 0, 0.8))
    plt.xlabel('Dzień tygodnia')
    plt.ylabel('Ilość filmów')
    plt.title(f'Ilość obejrzanych filmów na platformie YouTube w poszczególne dni tygodnia dla miesiąca {filter_month_name}')
    plt.xticks(rotation=45)  # Obrót etykiet osi X
    plt.tight_layout()  # Dopasowanie układu wykresu

    st.pyplot(plt.gcf())

#Wykres 3
# Tworzenie słownika, gdzie kluczem jest krotka (rok, miesiąc), a wartością ilość filmów
def update_month_plot():
    year_month_counts = {}
    for record in youtube_records:
        year = record['year']
        month = record['month']
        year_month = (year, month)
        year_month_counts[year_month] = year_month_counts.get(year_month, 0) + 1

    # Rozdzielenie danych na lata-miesiące i ilości
    year_months = list(year_month_counts.keys())
    counts = list(year_month_counts.values())

    # Konwersja krotek (rok, miesiąc) na czytelne napisy "YYYY-MM"
    year_month_labels = [f"{year}-{month:02d}" for year, month in year_months]

    # Tworzenie wykresu
    plt.figure(figsize=(10,6))
    plt.bar(year_month_labels[::-1], counts[::-1], color=(1, 0, 0, 0.8))
    plt.xlabel('Rok-Miesiąc')
    plt.ylabel('Ilość filmów')
    plt.title('Ilość obejrzanych filmów na platformie YouTube w poszczególnych miesiącach')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(plt.gcf())


#Wykres 4
def update_year_month_day_plot():
    # Tworzenie daty z rokiem, miesiącem i dniem
    year_month_day_dates = [f"{record['year']}-{record['month']:02d}-{record['day']:02d}" for record in youtube_records]

    # Obliczenie ilości obejrzanych filmów dla każdej daty
    date_counts = Counter(year_month_day_dates)

    # Tworzenie listy unikalnych dat w formacie rok-msc-dzień
    unique_dates = sorted(list(date_counts.keys()))

    # Ilości obejrzanych filmów dla poszczególnych dat
    view_counts = [date_counts[date] for date in unique_dates]

    # Tworzenie DataFrame z unikalnymi datami
    data = pd.DataFrame({'date': unique_dates, 'views': view_counts})
    data['year_month'] = data['date'].apply(lambda x: x[:7])

    # Unikalne pary rok-msc
    unique_year_month = data['year_month'].unique()

    # Definicja ręcznych kolorów dla poszczególnych miesięcy
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#8c564b', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728',
        '#9467bd', '#7f7f7f', '#e377c2', '#bcbd22', '#17becf',
        '#8c564b', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728',
        '#9467bd', '#7f7f7f', '#e377c2', '#bcbd22', '#17becf'
    ]

    # Tworzenie wykresu słupkowego
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data.index, data['views'], color=[color_palette[unique_year_month.tolist().index(ym)] for ym in data['year_month']])
    plt.xlabel('Data (RRRR-MM-DD)')
    plt.ylabel('Ilość filmów')
    plt.title('Ilość obejrzanych filmów na platformie YouTube w poszczególnych dniach')
    plt.xticks(range(0, len(unique_dates), 30), data['date'][::30], rotation=45)  # Wyświetlanie co 30 etykiet
    plt.tight_layout()

    st.pyplot(plt.gcf())

#Wykres 5
def top_channels_plot():
    # Utworzenie słownika dla zliczania filmów na każdym kanale
    channel_counter = Counter()

    # Iteracja po przefiltrowanych danych i zliczanie na każdym kanale
    for item in youtube_records:
        subtitles = item.get('subtitles', [])
        for subtitle in subtitles:
            channel_name = subtitle.get('name')
            channel_counter[channel_name] += 1

    # Rozbicie licznika na listy nazw kanałów i ilości filmów
    channel_names, video_counts = zip(*sorted(channel_counter.items(), key=lambda x: x[1], reverse=True))

    # Wybór top 10 kanałów z największą ilością filmów
    top_channels = channel_counter.most_common(10)
    channel_names, video_counts = zip(*top_channels)
    
    print(top_channels)

    # Tworzenie wykresu słupkowego
    plt.figure(figsize=(10, 6))
    plt.barh(channel_names, video_counts, color=(1, 0, 0, 0.8))
    plt.xlabel('Ilość filmów')
    plt.ylabel('Kanał')
    plt.title('Top 10 kanałów na YouTube z największą ilością obejrzanych filmów')
    #plt.xticks(rotation=45, ha='right') 

    plt.gca().invert_yaxis()  # Odwrócenie kolejności na osi Y

    st.pyplot(plt.gcf())

#wykres 6
def word_cloud():
    # Wyciąganie tytułów
    titles = [item['title'] for item in youtube_records]

    cleaned_titles = [title.lstrip('Watched').strip() for title in titles]

    # Tokenizacja, lematyzacja i usuwanie stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    words = []
    for title in cleaned_titles:
        tokens = word_tokenize(title.lower())
        lemmas = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        filtered_words = [word for word in lemmas if word not in stop_words]
        words.extend(filtered_words)

    # Tworzenie chmury słów
    wordcloud = WordCloud(width=800, height=410, background_color='white').generate(' '.join(words))

    # Wyświetlenie chmury słów
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Chmura słów na podstawie tytułów obejrzanych filmów na YouTube')
    
    st.pyplot(plt.gcf())



















# Ustawienia ogólne dla strony
st.set_page_config(
    page_title="Analiza danych - mój YouTube",
    layout="wide"
)

# Styl dla wyśrodkowanego tekstu
centered_text = '<p style="text-align:center;font-size:36px;">Analiza danych - mój YouTube</p>'
st.markdown('<h1 style="text-align:center;">Analiza danych - mój YouTube</h1>', unsafe_allow_html=True)

# Wybór wyświetlanego wykresu na podstawie zakładki
selected_tab = st.radio('Wybierz zbiór danych do wyświetlenia:', ['Youtube', 'Youtube Music']) #, 'Reklamy'])

# Domyślny wybór dla wyświetlenia wykresu
default_chart = 'Youtube'


# Wybór wyświetlanego wykresu na podstawie zakładki
if selected_tab == 'Youtube':
    st.markdown('<h3 style="text-align:center;">Podsumowanie filmowe - YouTube</h3>', unsafe_allow_html=True)
    
    #SEKCJA 1
    col3, col4 = st.columns(2)
    with col3:
        update_month_plot()

    with col4:
        update_year_month_day_plot()

    #SEKCJA 2
    col5, col6 = st.columns(2)

    with col5:
        top_channels_plot()
    
    with col6:
        word_cloud()

    #SEKCJA 3
    filter_month_name = st.selectbox('# Wybierz miesiąc', month_names)
    col1, col2 = st.columns(2)  
    
    with col1:
        update_plot(filter_month_name)
    
    with col2:
        update_day_of_week_plot(filter_month_name)

    # Wyświetl animowany wykres HTML za pomocą komponentu st.components.v1.html
    st.markdown('<h3>Top 10 Channels Over Time</h3>', unsafe_allow_html=True)
    html_file = open('top_channels_animation.html', 'r', encoding='utf-8').read()
    st.components.v1.html(html_file, width=1300, height=600)



elif selected_tab == 'Youtube Music':
    st.markdown('<h3 style="text-align:center;">Podsumowanie muzyczne - YouTube Music</h3>', unsafe_allow_html=True)
    st.markdown('<h5 style="text-align:center;">W budowie</h5>', unsafe_allow_html=True)
    #music_filter_month_name = st.selectbox('Wybierz miesiąc', month_names)
    #update_music_plot(music_filter_month_name)

# elif selected_tab == 'Reklamy':
#     st.markdown('<h3 style="text-align:center;">Analiza wyświetlonych reklam na YouTube</h3>', unsafe_allow_html=True)
#     ads_filter_month_name = st.selectbox('Wybierz miesiąc', month_names)
#     update_ads_plot(ads_filter_month_name)
