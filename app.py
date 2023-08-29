#Importowanie potrzebnych bibliotek
import streamlit as st
from collections import Counter
import locale

import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import nltk

from streamlit.components.v1 import html

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Ustawienia ogólne dla strony
st.set_page_config(
    page_title="Analiza danych - mój YouTube",
    layout="wide"
)

# # Odczytaj zbiory danych z plików JSON
# with open('youtube_records.json', 'r', encoding='utf-8') as file:
#     youtube_records = json.load(file)

# with open('youtube_music_records.json', 'r', encoding='utf-8') as file:
#     youtube_music_records = json.load(file)

# Load the data from JSON and use Python's caching mechanism
@st.cache_data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Load the data using the caching function
youtube_records = load_data('youtube_records.json')
youtube_music_records = load_data('youtube_music_records.json')


# Styl dla wyśrodkowanego tekstu
# centered_text = '<p style="text-align:center;font-size:36px;">Analiza danych - mój YouTube</p>'
st.markdown('<h1 style="text-align:center;">Analiza danych - mój YouTube</h1>', unsafe_allow_html=True)

# Wybór wyświetlanego wykresu na podstawie zakładki
selected_tab = st.radio('Wybierz zbiór danych do wyświetlenia:', ['Youtube', 'Youtube Music'])

# Domyślny wybór dla wyświetlenia wykresu
default_chart = 'Youtube'


# Wybór wyświetlanego wykresu na podstawie zakładki
if selected_tab == 'Youtube':
    st.markdown('<h3 style="text-align:center;">Podsumowanie filmowe - YouTube</h3>', unsafe_allow_html=True)
    
    #SEKCJA 1
    col1, col2 = st.columns(2)
    with col1:
        # Tworzenie słownika, gdzie kluczem jest krotka (rok, miesiąc), a wartością ilość filmów
        year_month_counts = {}
        for record in youtube_records:
            year = record['year']
            month = record['month']
            year_month = (year, month)
            year_month_counts[year_month] = year_month_counts.get(year_month, 0) + 1
    
        year_months = list(year_month_counts.keys())
        counts = list(year_month_counts.values())
    
        # Konwersja krotek na czytelne napisy "YYYY-MM"
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

    with col2:
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
    
        # Definicja kolorów dla poszczególnych miesięcy
        color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#8c564b', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728',
            '#9467bd', '#7f7f7f', '#e377c2', '#bcbd22', '#17becf',
            '#8c564b', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728',
            '#9467bd', '#7f7f7f', '#e377c2', '#bcbd22', '#17becf'
        ]
    
        # Tworzenie wykresu
        plt.figure(figsize=(10, 6))
        bars = plt.bar(data.index, data['views'], color=[color_palette[unique_year_month.tolist().index(ym)] for ym in data['year_month']])
        plt.xlabel('Data (RRRR-MM-DD)')
        plt.ylabel('Ilość filmów')
        plt.title('Ilość obejrzanych filmów na platformie YouTube w poszczególnych dniach')
        plt.xticks(range(0, len(unique_dates), 30), data['date'][::30], rotation=45)  # Wyświetlanie co 30 etykiet
        plt.tight_layout()
    
        st.pyplot(plt.gcf())

    #SEKCJA 2
    col3, col4 = st.columns(2)

    with col3:
        # Utworzenie słownika dla zliczania filmów na każdym kanale
        channel_counter = Counter()
    
        # Iteracja po przefiltrowanych danych i zliczanie na każdym kanale
        for item in youtube_records:
            subtitles = item.get('subtitles', [])
            for subtitle in subtitles:
                channel_name = subtitle.get('name')
                channel_counter[channel_name] += 1
    
        # Rozbicie licznika na listy nazw kanałów i ilości filmów
        #channel_names, video_counts = zip(*sorted(channel_counter.items(), key=lambda x: x[1], reverse=True))
    
        # Wybór top 10 kanałów z największą ilością filmów
        top_channels = channel_counter.most_common(10)
        channel_names, video_counts = zip(*top_channels)
    
        # Tworzenie wykresu słupkowego
        plt.figure(figsize=(10, 6))
        plt.barh(channel_names, video_counts, color=(1, 0, 0, 0.8))
        plt.xlabel('Ilość filmów')
        plt.ylabel('Kanał')
        plt.title('Top 10 kanałów na YouTube z największą ilością obejrzanych filmów')
    
        plt.gca().invert_yaxis()  
    
        st.pyplot(plt.gcf())
    
    with col4:
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
    
        # Wizualizacja
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Chmura słów na podstawie tytułów obejrzanych filmów na YouTube')
        
        st.pyplot(plt.gcf())

    #SEKCJA 3
    month_names = ['styczeń', 'luty', 'marzec', 'kwiecień', 'maj', 'czerwiec', 'lipiec', 'sierpień', 'wrzesień', 'październik', 'listopad', 'grudzień']
    filter_month_name = st.selectbox('# Wybierz miesiąc', month_names)
    col5, col6 = st.columns(2)  
    
    with col5:
        # Przygotowanie danych
        hour_counts = Counter((record['month'], record['hour']) for record in youtube_records)
        
        #month_names = ['styczeń', 'luty', 'marzec', 'kwiecień', 'maj', 'czerwiec', 'lipiec', 'sierpień', 'wrzesień', 'październik', 'listopad', 'grudzień']
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
    
    with col6:
        # Przetwarzanie danych
        day_of_week_counts = Counter()
    
        for record in youtube_records:
            # Konwertowanie daty i czasu z formatu "YYYY-MM-DD HH:MM:SS"
            datetime_str = record['time']
            datetime_obj = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            
            # Filtrowanie danych tylko dla wybranego miesiąca
            if datetime_obj.month == month_names.index(filter_month_name) + 1:
                day_name = datetime_obj.strftime('%A')
                
                day_of_week_counts[day_name] += 1
    
        days_of_week = list(day_of_week_counts.keys())
        view_counts = list(day_of_week_counts.values())
    
        data = pd.DataFrame({'Dzień tygodnia': days_of_week, 'Ilość filmów': view_counts})
    
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
        data['Dzień tygodnia'] = pd.Categorical(data['Dzień tygodnia'], categories=order, ordered=True)
    
        data = data.sort_values('Dzień tygodnia')
    
        filter_month = month_names.index(filter_month_name) + 1
    
        #Tworzenie wykresu
        plt.figure(figsize=(10, 5.55))
        plt.bar(data['Dzień tygodnia'], data['Ilość filmów'], color=(1, 0, 0, 0.8))
        plt.xlabel('Dzień tygodnia')
        plt.ylabel('Ilość filmów')
        plt.title(f'Ilość obejrzanych filmów na platformie YouTube w poszczególne dni tygodnia dla miesiąca {filter_month_name}')
        plt.xticks(rotation=45)  
        plt.tight_layout() 
    
        st.pyplot(plt.gcf())

    # Wyświetl animowany wykres HTML za pomocą komponentu st.components.v1.html
    st.markdown('<h3>Top 10 Channels Over Time</h3>', unsafe_allow_html=True)
    html_file = open('top_channels_animation.html', 'r', encoding='utf-8').read()
    st.components.v1.html(html_file, height=600, width=1300)


##########################################################################################################
#zakładka YT MUSIC

elif selected_tab == 'Youtube Music':
    st.markdown('<h3 style="text-align:center;">Podsumowanie muzyczne - YouTube Music</h3>', unsafe_allow_html=True)
    
    #WYKRES 1
    # Przygotowanie danych do DataFrame
    activity_counts_per_date = {}

    for record in youtube_music_records:
        timestamp = record['time']
        date = pd.to_datetime(timestamp).date()   # Wybieramy tylko datę, pomijając godzinę, minutę i sekundę
        
        if date in activity_counts_per_date:
            activity_counts_per_date[date] += 1
        else:
            activity_counts_per_date[date] = 1

    data = {'Date': list(activity_counts_per_date.keys()), 'ActivityCount': list(activity_counts_per_date.values())}
    df = pd.DataFrame(data)

    #Wykres
    st.markdown('<h3 style="font-size: 20px;">Filtr daty</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)  # Tworzenie dwóch kolumn

    with col1:
        start_date = st.date_input('Wybierz datę początkową', df['Date'].min())

    with col2:
        end_date = st.date_input('Wybierz datę końcową', df['Date'].max())

    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    fig = px.line(filtered_df, x='Date', y='ActivityCount', title='Aktywność na YouTube Music w czasie',
                labels={'Date': 'Data', 'ActivityCount': 'Liczba odsłuchanych utworów'}, color_discrete_sequence=['red'])

    # Wyświetlenie interaktywnego wykresu
    st.plotly_chart(fig, use_container_width=True)

    #WYKRES 2 - ARTYŚCI
    # Obliczenie rozkładu odsłuchań według artystów
    artist_counts = Counter()

    for record in youtube_music_records:
        if 'subtitles' in record and record['subtitles']:
            artist = record['subtitles'][0]['name']
            artist_counts[artist] += 1

    # Konwersja na DataFrame
    data_artist = {'Artist': [], 'Count': []}

    for artist, count in artist_counts.items():
        data_artist['Artist'].append(artist)
        data_artist['Count'].append(count)

    df_artists = pd.DataFrame(data_artist)

    # Wybór ilości artystów do uwzględnienia
    num_artists = st.slider('Wybierz ilość artystów', min_value=5, max_value=20, value=10)

    # Sortowanie artystów według liczby odsłuchań
    sorted_artists = df_artists.sort_values(by='Count', ascending=False).head(num_artists)

    # Tworzenie wykresu
    fig_artists = px.bar(sorted_artists, x='Count', y='Artist',
                        title=f'Top {num_artists} artystów z największą liczbą odsłuchań',
                        labels={'Count': 'Liczba odsłuchań', 'Artist': 'Artysta'},
                        color_discrete_sequence=['red'])

    fig_artists.update_layout(yaxis_tickfont=dict(size=10))

    # Wyświetlenie interaktywnego wykresu
    st.plotly_chart(fig_artists, use_container_width=True)


    # #WYKRES 3 - UTWORY
    # for record in youtube_music_records:
    #     title = record['title']

    #     # Sprawdzenie, czy pole 'subtitles' istnieje w rekordzie
    #     if 'subtitles' in record and record['subtitles']:
    #         subtitle = record['subtitles'][0]['name']  # Wybieramy nazwę wykonawcy z danych
    #         # Usunięcie 'VEVO' z końca nazwy wykonawcy
    #         subtitle = subtitle.replace('VEVO', '').strip()
    #         # Zmiana 'AmeliaMoore' na 'Amelia Moore'
    #         subtitle = subtitle.replace('AmeliaMoore', 'Amelia Moore')
    #         record['subtitles'][0]['name'] = subtitle  # Aktualizacja nazwy wykonawcy w rekordzie
    #     else:
    #         subtitle = 'Nieznany wykonawca'

    #     # Pobranie tylko części tekstu po ostatnim myślniku w tytule
    #     if '-' in title:
    #         title = title.rsplit('-', 1)[-1].strip()

    #     # Usunięcie tekstu w nawiasach na końcu tytułu
    #     if '(' in title and ')' in title:
    #         title = title.rsplit('(', 1)[0].strip()
        
    #     record['title'] = title


    # # Przygotowanie danych do analizy
    # song_counts = Counter()

    # for record in youtube_music_records:
    #     title = record['title']

    #     # Sprawdzenie, czy pole 'subtitles' istnieje w rekordzie
    #     if 'subtitles' in record and record['subtitles']:
    #         subtitle = record['subtitles'][0]['name']  # Wybieramy nazwę wykonawcy z danych
    #     else:
    #         subtitle = 'Nieznany wykonawca'

    #     song_counts[(title, subtitle)] += 1

    # # Konwersja na DataFrame
    # data = {'Title': [], 'Subtitle': [], 'Count': []}

    # for (title, subtitle), count in song_counts.items():
    #     data['Title'].append(title)
    #     data['Subtitle'].append(subtitle)
    #     data['Count'].append(count)

    # df_songs = pd.DataFrame(data)

    # # Wybór ilości utworów do uwzględnienia
    # num_songs = st.slider('Wybierz ilość utworów', min_value=5, max_value=20, value=10)

    # # Sortowanie utworów według liczby odsłuchań
    # sorted_df = df_songs.sort_values(by='Count', ascending=False).head(num_songs)

    # # Tworzenie wykresu
    # fig = px.bar(sorted_df, x='Count', y='Title', orientation='h',
    #             title=f'Top {num_songs} utworów z największą liczbą odsłuchań',
    #             labels={'Count': 'Liczba odsłuchań', 'Title': 'Tytuł utworu'}, color_discrete_sequence=['red'])

    # # Dodanie informacji o wykonawcy do etykiet na osi Y
    # fig.update_yaxes(ticktext=[f'{subtitle} - {title}' for subtitle, title  in zip(sorted_df['Subtitle'], sorted_df['Title'])],
    #                 tickvals=sorted_df['Title'])

    # fig.update_layout(yaxis_tickfont=dict(size=10))


    # # Wyświetlenie interaktywnego wykresu
    # st.plotly_chart(fig, use_container_width=True)




    @st.cache_data
    def prepare_song_data(records):
        # Przygotowanie danych do analizy
        song_counts = Counter()
    
        for record in records:
            title = record['title']
    
            # Sprawdzenie, czy pole 'subtitles' istnieje w rekordzie
            if 'subtitles' in record and record['subtitles']:
                subtitle = record['subtitles'][0]['name']  # Wybieramy nazwę wykonawcy z danych
            else:
                subtitle = 'Nieznany wykonawca'
    
            song_counts[(title, subtitle)] += 1
    
        # Konwersja na DataFrame
        data = {'Title': [], 'Subtitle': [], 'Count': []}
    
        for (title, subtitle), count in song_counts.items():
            data['Title'].append(title)
            data['Subtitle'].append(subtitle)
            data['Count'].append(count)
    
        df_songs = pd.DataFrame(data)
        return df_songs
    
    # Wywołanie funkcji i przypisanie wyniku do zmiennej
    df_songs_cached = prepare_song_data(youtube_music_records)
    
    # Wybór ilości utworów do uwzględnienia
    num_songs = st.slider('Wybierz ilość utworów', min_value=5, max_value=20, value=10)
    
    # Sortowanie utworów według liczby odsłuchań
    sorted_df = df_songs_cached.sort_values(by='Count', ascending=False).head(num_songs)
    
    # Tworzenie wykresu
    fig = px.bar(sorted_df, x='Count', y='Title', orientation='h',
                title=f'Top {num_songs} utworów z największą liczbą odsłuchań',
                labels={'Count': 'Liczba odsłuchań', 'Title': 'Tytuł utworu'}, color_discrete_sequence=['red'])
    
    # Dodanie informacji o wykonawcy do etykiet na osi Y
    fig.update_yaxes(ticktext=[f'{subtitle} - {title}' for subtitle, title  in zip(sorted_df['Subtitle'], sorted_df['Title'])],
                    tickvals=sorted_df['Title'])
    
    fig.update_layout(yaxis_tickfont=dict(size=10))
    
    # Wyświetlenie interaktywnego wykresu
    st.plotly_chart(fig, use_container_width=True)













    
    # #WYKRES 4
    # # Przygotowanie danych do analizy
    # hour_counts = np.zeros(24)

    # for record in youtube_music_records:
    #     timestamp = record['time']
    #     hour = pd.to_datetime(timestamp).hour
    #     hour_counts[hour] += 1

    # # Tworzenie wykresu polarowego 
    # fig_hourly_activity_polar = px.bar_polar(
    #     r=hour_counts,
    #     theta=[f'{hour:02}:00' for hour in range(24)],
    #     title='Godzinowa aktywność odsłuchiwania muzyki',
    #     labels={'theta': 'Godzina', 'r': 'Liczba odsłuchań'},
    #     start_angle=90,
    #     color_discrete_sequence=['red'],
    #     template='plotly_dark'  # Ustawienie ciemnego tła
    # )

    # # Dostosowanie osi theta i koloru wykresu
    # fig_hourly_activity_polar.update_traces(theta=[f'{hour:02}:00' for hour in range(24)])
    # fig_hourly_activity_polar.update_polars(hole=0.1, bgcolor='black')

    # #Wyświetlenie 
    # st.plotly_chart(fig_hourly_activity_polar, use_container_width=True)


    # Przygotowanie danych do analizy godzinowej aktywności
    @st.cache_data
    def prepare_hourly_activity_data(data):
        hour_counts = np.zeros(24)
    
        for record in data:
            timestamp = record['time']
            hour = pd.to_datetime(timestamp).hour
            hour_counts[hour] += 1
    
        return hour_counts
    
    # Generowanie wykresu polarowego na podstawie przygotowanych danych
    @st.cache_data
    def generate_polar_chart(hourly_activity_data):
        fig_hourly_activity_polar = px.bar_polar(
            r=hourly_activity_data,
            theta=[f'{hour:02}:00' for hour in range(24)],
            title='Godzinowa aktywność odsłuchiwania muzyki',
            labels={'theta': 'Godzina', 'r': 'Liczba odsłuchań'},
            start_angle=90,
            color_discrete_sequence=['red'],
            template='plotly_dark'
        )
    
        fig_hourly_activity_polar.update_traces(theta=[f'{hour:02}:00' for hour in range(24)])
        fig_hourly_activity_polar.update_polars(hole=0.1, bgcolor='black')
    
        return fig_hourly_activity_polar
    
    # Użycie przygotowanych danych i wygenerowanego wykresu
    hourly_activity_data = prepare_hourly_activity_data(youtube_music_records)
    polar_chart = generate_polar_chart(hourly_activity_data)
    
    # Wyświetlenie wykresu
    st.plotly_chart(polar_chart, use_container_width=True)


