import os
import sys
from music21 import *
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import seaborn as sns

from types import SimpleNamespace

import io
import base64

from nicegui import ui

################################
### POMOŽNE FUNKCIJE
################################

# Analiza izbrane datoteke
def xml_analysis(stream):

    elements = []
    for element in stream.recurse().getElementsByClass(['Note', 'Rest']):
        elements.append(element)
        
    metadata = extract_metadata(elements, stream)
    code = parson_code(elements)
    graph = contour(elements)
    lyrics = text.assembleAllLyrics(stream)
    
    return metadata, code, graph, lyrics

# Pretvornik za germansko in številčeno Helmholtz notacijo
def german_notation(p):
    if hasattr(p, 'name'):
        letter = p.name
    else:
        letter = p

    is_lower = letter[0].islower()
    upper_letter = letter.upper()

    if upper_letter == 'B':
        note_name = 'H'
    elif upper_letter == 'B-':
        note_name = 'B'
    else:
        note_name = upper_letter.replace('-', '\u266d')
    
    if hasattr(p, 'octave'):
        octave = int(p.octave)
        if octave >= 4:
            octave -= 3
            note_name = note_name.lower()
        elif octave == 3:
            octave = ''
            note_name = note_name.lower()
        elif octave == 2:
            octave = ''
        elif octave == 0:
            octave = 2
    else:
        octave = ''
    
    if is_lower:
        note_name = note_name.lower()
    return f'{note_name}{octave}'

# Pretvornik številske oznake (pitch space) v tradicionalno notacijo 
def num_to_note_name(num):

    note = pitch.Pitch()
    note.ps = num
    name = german_notation(note)
    
    return name

# Globalna spremenljivka za določanje širine stolpcev v intervalskem grafu
_min_duration = 1.0

###############################################
# NUMERIČNI PODATKI
# Funkcija ustvari seznam metapodatkov
# o danem dani podatkovni strukturi (Stream)
###############################################
def extract_metadata(elements, stream):

    metadata = dict()

    # Prepoznana tonaliteta
    key = stream.analyze('key')
    tonic = str(key).split(' ')[0]    
    key_name=f'{german_notation(tonic)}-{(key.mode).replace('minor', 'mol').replace('major', 'dur')}'
    metadata['key'] = key_name
    
    note_count = len(stream.flatten().notes)
    metadata['note_count'] = note_count

    # Notni obseg
    ambitus = analysis.discrete.Ambitus()
    min_pitch, max_pitch = ambitus.getPitchSpan(stream)
    amb_interval = ambitus.getSolution(stream)
    metadata['min_pitch'] = german_notation(min_pitch)
    metadata['max_pitch'] = german_notation(max_pitch)
    metadata['ambitus_interval'] = [amb_interval.name.replace('P', 'č').replace('M', 'v'), amb_interval.cents]    

    # Povprečna tonska višina ter razmerje med pozicijo najvišjega in najnižjega tona
    avg_pitch = pitch.Pitch()
    pitch_sum = 0
    count = 0
    first_note = None
    last_note = None
    position_max = None
    global _min_duration
    
    for element in elements: 
        if element.isNote: 
            if first_note is None:
                first_note = element.pitch.ps
            if position_max is None and element.pitch == max_pitch:
                position_max = 1
            elif position_max is None and element.pitch == min_pitch:
                position_max = 2
            last_note = element.pitch.ps
            pitch_sum += last_note
            count += 1
            
            dur = element.duration.quarterLength
            if dur > 0 and dur < _min_duration:
                _min_duration = dur
            

    avg_pitch.ps = pitch_sum/count
    metadata['avg_pitch'] = german_notation(avg_pitch)


    # Tip konture po Adamsovi tipologiji
    type = classify_adams_contour(first_note, last_note, max_pitch.ps, min_pitch.ps, position_max)
    metadata['Adams'] = type

    # Skupno trajanje (v četrtinkah)
    duration = stream.duration.quarterLength
    metadata['duration'] = int(duration)

    # Povprečna intervalna vrednost v skladbi
    avg_interval_num = features.jSymbolic.AverageMelodicIntervalFeature(stream).extract().vector
    avg_interval = interval.Interval(avg_interval_num[0])
    metadata['avg_interval'] = avg_interval.name.replace('P', 'č').replace('M', 'v')
    return metadata

####################################
### FUNKCIJE ZA PREDSTAVITEV KONTURE
####################################

### Adamsovi tipi kontur

#### TODO

CONTOUR_SHAPES_ADAMS = {
    "S1 D0 R0": [8, 2],
    "S2 D0 R0": [5, 5],
    "S3 D0 R0": [2, 8],
    
    "S1 D1 R1": [8, 10, 2],
    "S2 D1 R1": [5, 10, 5],
    "S3 D1 R1": [2, 10, 8],
    
    "S1 D1 R2": [8, 0, 2],
    "S2 D1 R2": [5, 0, 5],
    "S3 D1 R2": [2, 0, 8],
    
    "S1 D2 R1": [8, 10, 0, 2],
    "S2 D2 R1": [5, 10, 0, 5],
    "S3 D2 R1": [2, 10, 0, 8],
    
    "S1 D2 R2": [2, 0, 10, 8],
    "S2 D2 R2": [5, 0, 10, 5],
    "S3 D2 R2": [8, 0, 10, 2],
}

# Ikone za tip konture 
def get_contour_icon(type_str):

    y_values = CONTOUR_SHAPES_ADAMS.get(type_str, [5, 5]) 
    x_values = range(len(y_values))

    fig, ax = plt.subplots(figsize=(2, 1.5), layout='constrained')
    ax.plot(x_values, y_values, linewidth=2, marker='o', markersize=4) 
    ax.axis('off')
    ax.set_ylim(-1, 11)

    # Shrani graf s pretvorbo v base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)

    return f'data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}'

CONTOUR_IMAGE_CACHE = {t: get_contour_icon(t) for t in CONTOUR_SHAPES_ADAMS}

# Določitev tipa konture
def classify_adams_contour(I,F,H,L,pos_max):
    
    # Naklon konture (razmerje med prvo in zadnjo noto)
    if I > F:
        slope = 1
    elif I == F:
        slope = 2
    else: 
        slope = 3    

    # Deviacija naklona in recipročne vrednosti (razmerja med najvišjo in najnižjo noto)
    if H == max(I,F) and L == min(I,F): 
        deviation = 0
        reciprocal = 0
    elif L == min(I,F) or H == max(I,F):
        deviation = 1
        if H > I and H > F:
            reciprocal = 1
        else:
            reciprocal = 2
    else: 
        deviation = 2
        if pos_max == 1:
            reciprocal = 1
        else:
            reciprocal = 2

    return f'S{slope} D{deviation} R{reciprocal}'

### Parsonova koda
# * je vedno prvi znak
# U (up), če je nota višja od prejšnje
# R (repeat), če je nota enaka prejšnji
# D (down), če je nota nižja od prejšnje
def parson_code(list):
    
    code = str()
    previous = -1
    for element in list:
        if element.isNote:
            current = element.pitch.ps
            if previous == -1:              
                code += '*'        
            else:
                if current > previous:
                    code += 'U'
                elif current == previous:
                    code += 'R'
                elif current < previous:
                    code += 'D'
            previous = current

    return code

### Graf stroge konture
# če je znak nota, zabeležimo višino in trajanje
# če je znak pavza, zabeležimo le trajanje
def contour(element_list):
    points = []
    point_in_time = 0
    for element in element_list:
        if hasattr(element, 'pitch'):
            pitch = element.pitch.ps
        else: 
            pitch = np.nan            
        duration = element.duration.quarterLength
        
        points.append((pitch,point_in_time))
        point_in_time += duration
        points.append((pitch,point_in_time))

    df = pd.DataFrame(points, columns = ['pitch','time'])
    
    note_starts = df.iloc[::2].copy().reset_index(drop=True) # kjer je vsak element zapisan le kot ena (začetna) točka
    n = note_starts['pitch']
    p_value = note_starts['pitch'].values.astype(float)
    t_value = note_starts['time'].values.astype(float)
    mask = np.isfinite(t_value) & np.isfinite(p_value)
    x_clean = t_value[mask]
    y_clean = p_value[mask]

    ### Graf reducirane konture
    # Določajo jo lokalni ekstremi
    def reduced_contour():
        # Če je ton drugačen od naslednjega ali če je pavza
        is_different_pitch = (n != n.shift(1))
        is_rest = n.isna()

        selection = n[is_different_pitch | is_rest]
        
        # Označimo mesta, ki so "hrib" ali "dolina", torej kjer se graf prelomi
        notes_only = selection.dropna()
        previous = notes_only.shift(1)
        next = notes_only.shift(-1)
        is_peak = (notes_only > previous) & (notes_only > next)
        is_valley = (notes_only < previous) & (notes_only < next)
        turning_point = notes_only.index[is_peak | is_valley]

        # Ozančimo mejne note
        is_note = n.notna()
        is_offset = is_note & (n.shift(-1).isna())
        is_onset = is_note & (n.shift(1).isna())
        boundary_point = n.index[is_offset | is_onset]
        
        # Označimo pavze
        rest_point = n.index[is_rest]
 
        # Filter indeksov
        final = set(turning_point) | set(boundary_point) | set(rest_point)
                
        return note_starts.loc[sorted(final)]

    ### Polinomska krivulja
    def polynom_contour():
        poly_func = np.polynomial.Polynomial.fit(x_clean, y_clean, 3)
        x_smooth = np.linspace(x_clean.min(), x_clean.max(), 50)
        y_smooth = poly_func(x_smooth)
        
        return pd.DataFrame({'time': x_smooth, 'pitch': y_smooth})
                    
    ### Gaussovo filtriranje
    def gauss_contour():
        y_pitch = np.array(p_value, dtype=float)
        x_time = np.array(t_value, dtype=float)        
        is_note = np.isfinite(y_pitch)
        changes = np.diff(is_note.astype(int))
        split_indices = np.where(changes != 0)[0] + 1
        boundaries = np.concatenate(([0], split_indices, [len(y_pitch)]))

        line_x, line_y = [], []
        dots_x, dots_y = [], []
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            seg_y = y_pitch[start:end]
            seg_x = x_time[start:end]
            
            # Preskoči pavze
            if len(seg_y) == 0 or np.isnan(seg_y[0]):
                continue
                
            if len(seg_y) > 1:
                smooth_seg = gaussian_filter1d(seg_y, sigma=2.0)
                line_x.extend(seg_x)
                line_y.extend(smooth_seg)
                line_x.append(np.nan)
                line_y.append(np.nan)
            else:
                # Samo pika
                dots_x.extend(seg_x)
                dots_y.extend(seg_y)
                
        return SimpleNamespace(
                    line_x = np.array(line_x),
                    line_y = np.array(line_y),
                    dots_x = np.array(dots_x),
                    dots_y = np.array(dots_y)
                )
                        
    ### Stolpčni diagram intervalov
    def interval_points():
        intervals = note_starts['pitch'].diff()
        filter = np.isfinite(intervals)
        
        interval_size = intervals[filter]
        time = note_starts['time'][filter] - _min_duration

        df_i = pd.DataFrame({'intervals': interval_size, 'time': time})
        return df_i

    ### Toplotni zemljevid (Heatmap graph) 
    def heatmap_points():
        return pd.DataFrame({'time': x_clean,'pitch': y_clean})
    
        
    reduced_df = reduced_contour()
    interval_df = interval_points()
    polynom_df = polynom_contour()
    heatmap_df = heatmap_points()
    gauss_df = gauss_contour()

    return df, reduced_df, interval_df, polynom_df, heatmap_df, gauss_df # po želji: df <-> note_starts

#################################
### NALAGANJE PODATKOV IN ANALIZA
#################################

songs_data = []

def load_data():
    songs_data.clear()
    # Izbira datotek za analizo
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
    else:
        folder_name = './Testna_mapa'
    
    if not os.path.exists(folder_name):
        print("NAPAKA: Mapa ne obstaja.")
        return

    song_list = [f for f in os.listdir(folder_name) if f.endswith('.xml')]

    # Analiza vseh datotek s seznama
    for i, song in enumerate(song_list):
        file = os.path.join(folder_name, song)
        
        try: 
            stream = converter.parse(file)
            results = xml_analysis(stream.parts[0])
            title = f"{stream.metadata.title}{'\u200b' * i}" if stream.metadata.title else f"Neznano {i}"
            composer = stream.metadata.composer or 'Neznan avtor'
            subplot = results[2][0]            
            subplot_reduced = results[2][1]
            subplot_interval = results[2][2]
            subplot_polynom = results[2][3]
            subplot_heatmap = results[2][4]
            subplot_gauss = results[2][5]                                
            songs_data.append({
                'id': i,
                'title': title,
                'composer': composer,
                'metadata': results[0],
                'parson': results[1],
                'stroga': {'time': subplot.time, 'pitch': subplot.pitch},
                'reducirana': {'time': subplot_reduced.time, 'pitch': subplot_reduced.pitch},
                'interval': {'time': subplot_interval.time, 'intervals': subplot_interval.intervals},
                'polinom':{'time': subplot_polynom.time, 'pitch': subplot_polynom.pitch},
                'heatmap':{'time': subplot_heatmap.time, 'pitch': subplot_heatmap.pitch},
                'gauss':{'line_x': subplot_gauss.line_x,
                        'line_y': subplot_gauss.line_y,
                        'dots_x': subplot_gauss.dots_x,
                        'dots_y': subplot_gauss.dots_y},
                'lyrics': results[3],
                'show_stroga': False,
                'show_reducirana': True,
                'show_polinom': True,
                'show_gauss': False,
                'show_toplotni': False,
                'show_intervalni': True
            })
        except Exception as e:
            print(f'Napaka pri datoteki {song}: {e}')
    export_metadata(songs_data)
    print(f'Končano. Naloženih {len(songs_data)} skladb.')

# Shrani vse metapodatke v skupno xlsx datoteko
def export_metadata(songs):
    if not os.path.isdir('exports'):
        os.mkdir('exports')
        
    filename = 'exports/Metapodatki.xlsx'
    all_rows =[]
    column_mapping = {
            'key': 'Tonaliteta',
            'note_count': 'St. not',
            'duration': 'Trajanje',
            'min_pitch': 'Najnizji ton',
            'max_pitch': 'Najvisji ton',
            'ambitus_interval': 'Interval ambitusa',
            'avg_pitch': 'Povprečna tonska višina',
            'avg_interval': 'Povprečni interval',
        }
    for song in songs: 
        row = {'ID': song.get('id'),'Naslov': song.get('title'), 'Skladatelj': song.get('composer')}
        meta_data = song['metadata']
        row['ambitus_interval'] = meta_data['ambitus_interval'][0]
        row.update(meta_data)
        row['Parsonova koda'] = song.get('parson'),
        row['Besedilo'] = song.get('lyrics')

        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.rename(columns=column_mapping, inplace=True)
    df.to_excel(filename, index=False)


#########################################
#########################################
### GRAFIČNI UPORABNIŠKI VMESNIK
#########################################
#########################################

@ui.page('/')
def main_page():
    
    PAGE_SIZE = 12
    state = {
        'page': 0,
        'comparison_ids': set(),
        'comparison_mode': 'reducirana',
        'global_mode': 'both',
        'adams_filter': None,
        'sort_key': 'alphabet',
        'sort_order': 'asc',
        
        'show_stroga': False,
        'show_reducirana': True,
        'show_polinom': True,
        'show_gauss': False,
        'show_toplotni': False,
        'show_intervalni': True,

    }
    sort_options = {
        'alphabet': 'Abecedi',
        'adams': 'Tipu konture',
        'key': 'Tonaliteti',
        'ambitus': 'Ambitusu',
        'duration': 'Trajanju',
    }
    
    ############################
    ### POMOŽNE FUNKCIJE
    ############################
    
    def draw_plot(song, ax, ax_1):
        if song['show_stroga']:
            ax.plot(song['stroga']['time'], song['stroga']['pitch'], c='blue', label='Stroga', alpha=0.8)
        if song['show_reducirana']:
            ax.plot(song['reducirana']['time'], song['reducirana']['pitch'], c='darkorange', label='Reducirana', alpha=0.8)
        if song['show_polinom']:
            ax.plot(song['polinom']['time'], song['polinom']['pitch'], color='purple', linestyle='-', alpha=0.8, label='Polinomska')
        if song['show_toplotni']:
            sns.kdeplot(x=song['heatmap']['time'], y=song['heatmap']['pitch'], cmap="GnBu", fill=True, ax=ax, alpha=0.6, thresh=0.05, bw_adjust=0.5)
        
        ### Gaussova krivulja
        # glajenje konture z Gaussovim filtrom
        if song['show_gauss']:
            gauss_data = song['gauss']
            if len(gauss_data['line_x']) > 0:
                sns.lineplot(x=gauss_data['line_x'], y=gauss_data['line_y'], ax=ax, linewidth=2, color='green', label='Gauss')

            if len(gauss_data['dots_x']) > 0:
                dot_label = 'Gauss' if len(gauss_data['line_x']) == 0 else None
                sns.lineplot(x=gauss_data['dots_x'], y=gauss_data['dots_y'], ax=ax, marker='o', color='green', label=dot_label)

        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Tonska višina")
        ax.set_xlabel("Čas")
        ax.set_xlim(left=-0.5, right=max(song['stroga']['time'] + 0.5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: num_to_note_name(x)))
        if not song['show_toplotni'] and ax.lines: 
            ax.legend()
        
        
        ### intervalni graf
        if song['show_intervalni']:
            colors = ['blue' if i > 0 else 'orange' for i in song['interval']['intervals']]
            ax_1.bar(song['interval']['time'], song['interval']['intervals'], color=colors, width=_min_duration, align='edge')

            ax_1.set_ylabel("Interval (poltoni)")
            ax_1.set_xlabel("Čas")
            ax_1.axhline(0, color='gray', linewidth=1)
            ax_1.set_ylim(-8, 8)
            ax_1.grid(True, alpha=0.3)

    
    image_cache = {}
    plot_cache = {}
    # Generiranje slik grafov 
    def create_plot_image(song):
        cache_key = (
            song.get('title', 'unknown'),
            song.get('show_intervalni', False),
            song.get('show_stroga', False), 
            song.get('show_reducirana', False), 
            song.get('show_polinom', False),
            song.get('show_toplotni', False),
            song.get('show_gauss', False)
        )
        if cache_key in image_cache:
            return image_cache[cache_key]
        
        show_intervals = song.get('show_intervalni', False)
        rows = 2 if show_intervals else 1
        height = 9 if show_intervals else 6
        ratio = {'height_ratios':[2,1]} if show_intervals else {}
        
        fig, axes = plt.subplots(nrows=rows, ncols=1, figsize=(12, height), 
                                  sharex=True, layout='constrained', gridspec_kw = ratio)
                
        ax = axes[0] if show_intervals else axes
        ax_1 = axes[1] if show_intervals else None

        draw_plot(song, ax, ax_1)

        # Shranjevanje v bralni pomnilnik
        buf = io.BytesIO()
        fig.savefig(buf, format = 'png', bbox_inches = 'tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8') 
        final_result = f'data:image/png;base64,{img_str}'
        image_cache[cache_key] = final_result

        return final_result

    # Interaktivni graf v plotly
    def create_interactive_plot(song):

        cache_key = (
        song.get('title', 'unknown'),
        song.get('show_intervalni', False),
        song.get('show_stroga', False), 
        song.get('show_reducirana', False), 
        song.get('show_polinom', False),
        song.get('show_toplotni', False),
        song.get('show_gauss', False)
        )
        if cache_key in plot_cache:
            return plot_cache[cache_key]
    
        show_intervals = song.get('show_intervalni', False)
        rows = 2 if show_intervals else 1
        height = [0.7, 0.3] if show_intervals else [1.0]
        
        fig = make_subplots(rows=rows, cols=1, row_heights=height, shared_xaxes=True, vertical_spacing=0.05)

        if song.get('show_stroga'):
            fig.add_trace(go.Scatter(x=song['stroga']['time'], y=song['stroga']['pitch'], mode='lines+markers', name='Stroga', line=dict(color='blue', width=2),opacity=0.8), row=1, col=1)
        if song.get('show_reducirana'):
            fig.add_trace(go.Scatter(x=song['reducirana']['time'], y=song['reducirana']['pitch'], mode='lines', name='Reducirana', line=dict(color='darkorange', width=2), opacity=0.8), row=1, col=1)        
        if song.get('show_polinom'):
            fig.add_trace(go.Scatter(x=song['polinom']['time'], y=song['polinom']['pitch'], mode='lines', name='Polinomska', line=dict(color='purple', width=2), opacity=0.8), row=1, col=1)
        if song.get('show_gauss'):
            g = song['gauss']        
            if len(g.get('line_x', [])) > 0:
                fig.add_trace(go.Scatter(x=g['line_x'], y=g['line_y'],mode='lines', name='Gauss', line=dict(color='green', width=2), opacity=0.8), row=1, col=1)
            if len(g.get('dots_x', [])) > 0:
                dot_label = 'Gauss' if len(g['line_x']) == 0 else None
                fig.add_trace(go.Scatter(x=g['dots_x'], y=g['dots_y'], mode='markers', name=dot_label, marker=dict(color='green', size=6), opacity=0.8, showlegend=False), row=1, col=1)                
        if song.get('show_toplotni'):
            x_val = np.array(song['heatmap']['time'])
            y_val = np.array(song['heatmap']['pitch'])
            
            if len(x_val) > 1:                
                x_min, x_max = x_val.min(), x_val.max()
                y_min, y_max = y_val.min(), y_val.max()
                
                x_buf = (x_max - x_min) * 0.1
                y_buf = (y_max - y_min) * 0.1
                                
                xi = np.linspace(x_min - x_buf, x_max + x_buf, 100)
                yi = np.linspace(y_min - y_buf, y_max + y_buf, 100)                
                xi_grid, yi_grid = np.meshgrid(xi, yi)
                                
                positions = np.vstack([xi_grid.ravel(), yi_grid.ravel()])
                values = np.vstack([x_val, y_val])
                
                kernel = stats.gaussian_kde(values)                
                kernel.set_bandwidth(bw_method=kernel.factor * 0.5)
                
                zi = np.reshape(kernel(positions).T, xi_grid.shape)                                
                fig.add_trace(go.Contour(x=xi, y=yi, z=zi, colorscale='GnBu', opacity=0.6, showscale=False, 
                                         contours=dict(start=zi.max() * 0.05, end=zi.max(),size=zi.max() * 0.05, showlines=False,coloring='fill'),
                                         hoverinfo='skip', name='Toplotni'), row=1, col=1)

        if show_intervals:
            intervals = song['interval']['intervals']            
            colors = ['blue' if i > 0 else 'orange' for i in intervals]            
            fig.add_trace(go.Bar(x=song['interval']['time'], y=intervals, marker_color=colors, showlegend=False), row=2, col=1)                
                                    
            fig.update_yaxes(title_text="Interval (poltoni)", range=[-9, 9], tickvals=[-8, -6, -4, -2, 0, 2, 4, 6, 8], row=2, col=1)
            fig.update_xaxes(title_text="Čas", showgrid=True, row=2, col=1)
            fig.add_hline(y=0, line_width=1, line_color="gray", row=2, col=1)
        else: 
            fig.update_xaxes(title_text="Čas", showgrid=True, row=1, col=1)

        fig.update_layout(template='plotly_white', height=600, hovermode="x unified", margin=dict(l=50, r=20, t=40, b=40), legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))

        all_pitches = []
        all_pitches.extend(song['stroga']['pitch'])        

        valid_pitches = [p for p in all_pitches if p is not None and p == p]                
        min_p, max_p = min(valid_pitches), max(valid_pitches)
        tick_vals = list(range(int(min_p)-1, int(max_p)+2))
        tick_text = [num_to_note_name(p) for p in tick_vals]
                
        fig.update_yaxes(title_text='Tonska višina',tickvals=tick_vals,ticktext=tick_text,gridcolor='rgba(0,0,0,0.1)', row=1, col=1)
        fig.update_xaxes(gridcolor='rgba(0,0,0,0.1)', row=1, col=1, range=[-0.5, max(song['stroga']['time']) + 0.5])
        
        plot_cache[cache_key] = fig
        return fig


    # Nastavitve spustnega menija
    def get_sort_value(song, key):
        
        if state['sort_order'] == 'desc':
            is_compared = 1 if song['id'] in state['comparison_ids'] else 0
        else:
            is_compared = 0 if song['id'] in state['comparison_ids'] else 1
        
        if key == 'alphabet':
            result = song['title'].lower()
        elif key == 'adams':
            result = song['metadata']['Adams']
        elif key == 'ambitus':
            result = song['metadata']['ambitus_interval'][1]
        elif key == 'duration':
            result = song['metadata']['duration']
        elif key == 'key':
            result = song['metadata']['key']

        return (is_compared, result)
        
    # Izbira skladb za primerjanje
    def toggle_comparison(song_id, is_checked):
        if is_checked:
            state['comparison_ids'].add(song_id)
        else:
            state['comparison_ids'].discard(song_id)
        render_comparison_plot.refresh()
    
    # Primerjanje kontur
    def update_comparison_mode(mode):
        state['comparison_mode'] = mode
        render_comparison_plot.refresh()

    # Vklop/izklop subplotov za vse grafe naenkrat
    def toggle_global_view(view_type):
        key = f'show_{view_type}'
        state[key] = not state.get(key, False)
        
        for song in songs_data:
            song[key] = state[key]
        
        render_dashboard.refresh()
    
    # Zapiranje primerjalnega grafa
    def clear_comparison():
        state['comparison_ids'].clear()
        state['comparison_mode'] = 'reducirana'
        render_comparison_plot.refresh()
        render_dashboard.refresh()

    # Filtriranje in sortiranje kartic
    def get_processed_data():
        # Filtriranje
        if state['adams_filter']:
            data = [s for s in songs_data if s['metadata']['Adams'] == state['adams_filter']]
        else:
            data = list(songs_data)
        # Sortiranje
        data.sort(key=lambda s: get_sort_value(s, state['sort_key']),reverse=(state['sort_order'] == 'desc'))
        
        return data

    # Številčenje strani
    def change_page(delta, max_page_index):
        new_page = state['page'] + delta
        if 0 <= new_page <= max_page_index:
            state['page'] = new_page
            ui.run_javascript('window.scrollTo({ top: 0, behavior: "smooth" })')
            render_dashboard.refresh()

    # Posodobitev strani za sortiranje in filtriranje
    def update_view(key=None, value=None, order_toggle=False):
        if key == 'filter': 
            state['adams_filter'] = value
            state['page'] = 0
        elif key == 'sort':
            state['sort_key'] = value
        if order_toggle:
            state['sort_order'] = 'desc' if state['sort_order'] == 'asc' else 'asc'
        render_dashboard.refresh()

    # Pretvorba podatkov v xlsx tabelo
    def export_graph_data_to_excel(song):        
        filename = f"exports/{song['title']}_podatki.xlsx"
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:

            pd.DataFrame(song['stroga']).to_excel(writer, sheet_name='Stroga', index=False)
            pd.DataFrame(song['reducirana']).to_excel(writer, sheet_name='Reducirana', index=False)
            pd.DataFrame(song['polinom']).to_excel(writer, sheet_name='Polinomska', index=False)
            g = song['gauss']
            df_line = pd.DataFrame({'time': g.get('line_x', []), 'pitch': g.get('line_y', []), 'l/d': 'line'})
            df_dots = pd.DataFrame({'time': g.get('dots_x', []), 'pitch': g.get('dots_y', []), 'l/d': 'dot'})
            df_gauss = pd.concat([df_line, df_dots], axis=0)
            df_gauss = df_gauss.sort_values(by='time')
            df_gauss.to_excel(writer, sheet_name='Gaussova', index=False)
            
        print(f"Datoteka shranjena v: {os.path.abspath(filename)}")
        with open(filename, 'rb') as f:
            return f.read()

    # Shranjevanje xlsx tabele
    def handle_graph_data_export(song):
        excel_data = export_graph_data_to_excel(song)
        ui.download(excel_data, filename=f"{song['title']}_tocke_grafov.xlsx")
    
    ######################################################
    # GRADNIKI PRIKAZOVALNIKA
    ######################################################

    # Sestavi kartico za dano skladbo
    def render_song_card(song):
        with ui.card().classes('w-full p-2'):
            with ui.row().classes('w-full justify-between items-start no-wrap'):
                with ui.column().classes('pr-2 flex-1 gap-0'):
                    ui.label(song['title'].upper()).classes('font-bold text-lg break-words')
                    ui.label(song['composer']).classes('text-xs')
                ui.checkbox(value=(song['id'] in state['comparison_ids']),
                            on_change=lambda e, s=song['id']: (toggle_comparison(s, e.value), update_view())).tooltip('Dodaj v primerjalni graf')
            
            # Prikaz metapodatkov in ostalih tekstovnih podatkov
            info_text = (f'Tonaliteta: {song['metadata']['key']}* \n'
                         f'Št. not: {song['metadata']['note_count']} | Trajanje: {song['metadata']['duration']}\u2669 \n'
                         f'Ambitus: {song['metadata']['min_pitch']} < {song['metadata']['max_pitch']} (interval:\u00A0{song['metadata']['ambitus_interval'][0]})\n'
                         f'Povp. tonska višina: {song['metadata']['avg_pitch']} | Povp.\u00A0interval:\u00A0{song['metadata']['avg_interval']}')
            
            with ui.row().classes('w-full items-start justify-between no-wrap'):
                ui.label(info_text).classes('text-xs text-slate-500 whitespace-pre-line mb-1 h-20')
                ui.image(CONTOUR_IMAGE_CACHE.get(song['metadata']['Adams'])).classes('w-24 h-16 object-contain')
                ui.label(f'Tip konture: \n {song["metadata"]["Adams"]}').classes('text-sm text-slate-500 font-bold whitespace-pre-line text-right')

            with ui.row():
                with ui.row().classes('items-center gap-1 cursor-pointer inline-flex'):
                    ui.icon('lyrics').classes('text-slate-500 text-sm')
                    ui.label('BESEDILO').classes('text-xs font-bold text-slate-500')
                    with ui.menu().classes('bg-white p-4 shadow-xl border'):
                        ui.label(song["lyrics"]).classes('bg-white text-gray-900 text-xs whitespace-pre-line max-w-sm z-50 select-text')
                
                with ui.row().classes('items-center gap-1 cursor-pointer inline-flex'):
                    ui.icon('data_array').classes('text-slate-500 text-sm')
                    ui.label('PARSONOVA KODA').classes('text-xs font-bold text-slate-500')
                    with ui.menu().classes('bg-white p-4 shadow-xl border'):
                        ui.label(song["parson"]).classes('break-all text-xs max-w-[200px] select-text')
                    
            # Prikaz grafov
            container = ui.column().classes('w-full items-center justify-center')
            with container:                                
                initial_src = create_plot_image(song)

                # Interaktivni prikazovalnik grafov                                    
                with ui.dialog() as large_dialog, ui.card().classes('w-full max-w-6xl h-[80vh] p-0'):                    
                    with ui.row().classes('w-full justify-between p-2 bg-slate-100 border-b'):
                        ui.label(song['title']).classes('text-lg font-bold')
                        ui.button(icon='close', on_click=large_dialog.close).props('flat round dense')                                        
                    plot_container = ui.column().classes('w-full h-full p-4')
                
                def open_interactive_view():
                    plot_container.clear()
                    with plot_container:                                                                        
                        fig = create_interactive_plot(song)
                        ui.plotly(fig).classes('w-full h-full')
                       
                    large_dialog.open()

                # Statični prikaz/predogled
                graph_image = ui.image(initial_src)\
                    .classes('w-full max-w-[800px] cursor-pointer')\
                    .on('click', open_interactive_view)
                                
                def update_graph():                
                    new_src = create_plot_image(song)
                    graph_image.set_source(new_src)
                
                with ui.row().classes('w-full justify-between mt-2'):                                                                                
                    ui.button('XLSX', icon='download', on_click=lambda: handle_graph_data_export(song)) \
                        .props('flat dense color=grey').classes('text-xs')
                                        
                    with ui.button('Sloji', icon='layers').props('flat dense color=grey').classes('text-xs'):
                        with ui.menu().classes('p-3 bg-white border rounded shadow-lg'):
                            def add_layer_switch(label, key):
                                ui.switch(label, value=song.get(key, False), 
                                        on_change=lambda e: (
                                            song.update({key: e.value}),
                                            update_graph()
                                        )).props('dense')
                                
                            with ui.column().classes('gap-1'):
                                add_layer_switch('Stroga', 'show_stroga')
                                add_layer_switch('Reducirana', 'show_reducirana')
                                ui.separator()
                                add_layer_switch('Polinom', 'show_polinom')
                                add_layer_switch('Gauss', 'show_gauss')
                                add_layer_switch('Toplotni graf', 'show_toplotni')
                                ui.separator()
                                add_layer_switch('Intervali', 'show_intervalni')
                                
                                
    # Primerjalni graf
    @ui.refreshable
    def render_comparison_plot():
        
        container = ui.column().classes('w-full sticky top-0 z-50 p-4 bg-white/80 backdrop-blur-md border-b border-slate-200 shadow-sm mb-4')
        if not state['comparison_ids'] or len(state['comparison_ids']) < 1:
            container.set_visibility(False)
            return
        
        container.set_visibility(True)
                
        with container:
            def create_comparison_fig():
                fig = go.Figure()
                mode = state['comparison_mode']                                
                all_pitches = []
                                
                for i, song_id in enumerate(state['comparison_ids']):
                    song = next((s for s in songs_data if s['id'] == song_id), None)                                        
                    title = song['title']                                        
                    if mode == 'gauss':
                        g = song['gauss']
                        group_name = f"{title.capitalize()}"
                        palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
                        song_color = palette[i % len(palette)] 
                        if len(g.get('line_x', [])) > 0:
                            fig.add_trace(go.Scatter(x=g['line_x'], y=g['line_y'],mode='lines', name=group_name, line=dict(color=song_color, width=2), opacity=0.8))
                            all_pitches.extend(g['line_y'])
                        if len(g.get('dots_x', [])) > 0:
                            show_label = True if len(g.get('line_x', [])) == 0 else False
                            fig.add_trace(go.Scatter(x=g['dots_x'], y=g['dots_y'], mode='markers', name=group_name if show_label else None, 
                                                     legendgroup=group_name, marker=dict(color=song_color, size=6), showlegend=show_label, opacity=0.8))      
                            all_pitches.extend(g['dots_y'])
                    else:
                        lines = 'lines+markers' if mode=='stroga' else 'lines'
                        fig.add_trace(go.Scatter(x=song[mode]['time'], y=song[mode]['pitch'], mode=lines, name=title.capitalize(), opacity=0.8, line=dict(width=2)))
                        all_pitches.extend(song[mode]['pitch'])
                
                valid_pitches = [p for p in all_pitches if p is not None and p == p]                
                min_p, max_p = min(valid_pitches), max(valid_pitches)
                tick_vals = list(range(int(min_p)-1, int(max_p)+2))
                tick_text = [num_to_note_name(p) for p in tick_vals]
            
                fig.update_layout(template='plotly_white', margin=dict(l=40, r=20, t=10, b=20), height=300, legend=dict(orientation="h", y=1.1, x=0),hovermode="x unified")                
                fig.update_yaxes(title_text="Višina tona", tickvals=tick_vals, ticktext=tick_text, gridcolor='rgba(0,0,0,0.1)')
                fig.update_xaxes(title_text="Čas (četrtinke)", gridcolor='rgba(0,0,0,0.1)')
                
                return fig
            

            with ui.row().classes('w-full items-center justify-center relative mb-1'):
                ui.label('Primerjava izbranih skladb').classes('text-xl font-bold text-slate-700')
                ui.button(icon='close', on_click=clear_comparison).props('flat round dense color=grey').classes('absolute right-0 top-0').tooltip('Zapri primerjavo in počisti izbor')
            
            fig = create_comparison_fig()
            ui.plotly(fig).classes('w-2/3 h-[350px] self-center')

            with ui.row().classes('w-full justify-center'):
                ui.toggle(
                    options={'reducirana': 'reducirana kontura', 'stroga': 'stroga kontura', 'polinom': 'Polinomska', 'gauss':'Gaussova'},
                    value=state['comparison_mode'],
                    on_change=lambda e: update_comparison_mode(e.value)
                )

    # Gumbi za filtriranje
    @ui.refreshable
    def render_filter_buttons():
        btn_base = 'px-4 py-1 rounded-full border'
        btn_inactive = 'bg-white text-slate-500 border-gray-300 hover:bg-gray-100'
        
        def create_btn(label, key, active_color_class, tooltip=None):
            is_active = state.get(f'show_{key}')
            current_style = active_color_class if is_active else btn_inactive

            btn = ui.button(label, on_click=lambda: (toggle_global_view(key), render_filter_buttons.refresh())) \
                .props('flat') \
                .classes(f'{btn_base} {current_style}')
            if tooltip:
                with btn:
                    ui.tooltip(tooltip).classes('bg-gray text-white text-xs')
        with ui.row().classes('items-center gap-2'):
            create_btn('Strogi', 'stroga', 'bg-blue-600 text-white border-blue-700') 
            create_btn('Reducirani', 'reducirana', 'bg-orange-500 text-white border-orange-600') 
            create_btn('Gauss', 'gauss', 'bg-green-600 text-white border-green-700') 
            create_btn('Polinom', 'polinom', 'bg-purple-600 text-white border-purple-700')
            create_btn('Intervalni', 'intervalni', 'bg-slate-600 text-white border-slate-700')
            create_btn('Toplotni', 'toplotni', 'bg-teal-600 text-white border-teal-700', tooltip="Lahko traja nekaj sekund")
        
    ####################
    # OSREDNJI RENDERER
    ####################
    @ui.refreshable
    def render_dashboard():
        
        processed_list = get_processed_data()
        total_items = len(processed_list)
        max_page_index = max(0, (total_items - 1) // PAGE_SIZE)
        
        start_index = state['page'] * PAGE_SIZE
        end_index = start_index + PAGE_SIZE
        current_batch = processed_list[start_index:end_index]
        
        ui.label('Pregledovalnik melodičnih kontur').classes('text-2xl font-bold mt-4 px-4')
        
        render_comparison_plot() 

        with ui.row().classes('w-full items-center justify-between p-4 bg-slate-50 border-b-4 border-blue-100'):
            with ui.row().classes('items-center gap-4'):
                ui.label('Prikaži grafe:').classes('mr-2 font-bold text-slate-500')
                render_filter_buttons()
                
                
            ui.label(f'Št. skladb: {total_items}').classes('text-slate-500 text-sm font-bold')
            with ui.row().classes('items-center gap-4'):
                # spustni meni za filtriranje po tipu kontur (Adams)
                with ui.button(icon='filter_alt').classes('mx-4'):
                    button_text = state['adams_filter'] if state['adams_filter'] else 'Filter kontur'
                    ui.label(button_text).classes('ml-2')
                    
                    with ui.menu().classes('p-2') as contour_menu:

                        with ui.row().classes('w-full items-center justify-between mb-2 px-2'):
                            ui.label('Izbira konture').classes('text-xs font-bold text-gray-400 uppercase')
                            if state['adams_filter']:
                                ui.button(icon='close', on_click=lambda: (update_view('filter', None), contour_menu.close())) \
                                    .props('flat size=sm color=red')
                                
                        with ui.grid(columns=3).classes('gap-2 p-1'):
                            for label in CONTOUR_SHAPES_ADAMS.keys():
                                is_selected = (state['adams_filter'] == label)
                                img_src = CONTOUR_IMAGE_CACHE.get(label)
                                
                                with ui.card().classes(f'''
                                    cursor-pointer p-2 transition-all border-2 w-24 h-20
                                    items-center justify-center overflow-hidden
                                    {'border-blue-500 bg-blue-50' if is_selected else 'border-slate-100 hover:border-blue-200'}
                                ''').on('click', lambda _, l=label: (update_view('filter', l), contour_menu.close())):
                                    
                                    ui.image(img_src).classes('w-16 h-10 object-contain pointer-events-none')
                    
                # spustni meni za sortiranje
                ui.select(options=sort_options, value=state['sort_key'], label='Razvrsti po',on_change=lambda e: update_view('sort', e.value)).classes('w-48').props('dense')

                # smer sortiranja
                icon_name = 'arrow_upward' if state['sort_order'] == 'asc' else 'arrow_downward'
                ui.button(icon=icon_name, 
                        on_click=lambda: update_view(order_toggle=True)).props('flat round dense')
            
        # Navigacija po straneh zgoraj
        if total_items > PAGE_SIZE:
            with ui.row().classes('w-full justify-center items-center gap-4 mb-2'):
                back_button = ui.button(icon='chevron_left', on_click=lambda: change_page(-1, max_page_index)).props('flat')
                if state['page'] <= 0: 
                    back_button.set_visibility(False)
                ui.label(f'Stran {state['page'] + 1} / {max_page_index + 1}')
                next_button = ui.button(icon='chevron_right', on_click=lambda: change_page(1, max_page_index)).props('flat')
                if state['page'] >= max_page_index: 
                    next_button.set_visibility(False)
        
        
        # Če nobena skladba ne ustreza filtru
        if total_items == 0:
            ui.label('Izbranemu tipu konture ne ustreza nobena skladba.').classes('w-full text-center text-slate-500 mt-10')
            return

        # Mreža kartic
        if total_items == 0:
            ui.label('Ni rezultatov.').classes('w-full text-center text-slate-500 mt-10')
        else:
            with ui.grid().classes('w-full grid-cols-1 md:grid-cols-2 l:grid-cols-3 xl:grid-cols-4 gap-4 items-start'):
                for song in current_batch:
                    render_song_card(song)

        # Navigacija po straneh na dnu
        if total_items > PAGE_SIZE:
            with ui.row().classes('w-full justify-center items-center gap-4 mt-4'):
                back_button = ui.button(icon='chevron_left', on_click=lambda: change_page(-1, max_page_index)).props('flat')
                if state['page'] <= 0: 
                    back_button.set_visibility(False)
                ui.label(f'Stran {state['page'] + 1} / {max_page_index + 1}')
                next_button = ui.button(icon='chevron_right', on_click=lambda: change_page(1, max_page_index)).props('flat')
                if state['page'] >= max_page_index: 
                    next_button.set_visibility(False)

        ui.label('Opomba: Znak ~ označuje četrttonsko spremembo navzgor, znak ` pa četrttonsko spremembo navzdol.\n' 
                 '* Tonaliteta je analizirana po algoritmu Krumhansl-Schmuckler iz knjižnice Music21').classes('text-xs text-slate-500 whitespace-pre-line mb-1')
        
    load_data()
    render_dashboard()

#######################################
### ZAŽENI GRAFIČNI UPORABNIŠKI VMESNIK
#######################################
if __name__ in {'__main__', '__mp_main__'}:
     ui.run(native = False, reload = True, title = 'Pregledovalnik melodičnih kontur')