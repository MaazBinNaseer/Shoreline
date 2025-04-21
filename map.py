import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import folium
from folium.plugins import Fullscreen
from folium.features import GeoJsonTooltip
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO


# === Utility: Clean for Folium ===
def clean_for_folium(gdf):
    keep_cols = [col for col in gdf.columns if gdf[col].dtype.kind in "biufcO" or col == "geometry"]
    drop_cols = [col for col in gdf.columns if "datetime" in str(gdf[col].dtype).lower() or "date" in str(gdf[col].dtype).lower()]
    return gdf[keep_cols].drop(columns=drop_cols, errors='ignore')

# === NSM Color Classification (shared across both layers) ===
def get_fixed_nsm_color(nsm):
    if nsm <= -80:
        return '#800026'  # Deep red (strong erosion)
    elif -80 < nsm <= -40:
        return '#BD0026'  # Red
    elif -40 < nsm <= -20:
        return '#E31A1C'  # Orange-red
    elif -20 < nsm <= -5:
        return '#FC4E2A'  # Orange
    elif -5 < nsm < 0:
        return '#FD8D3C'  # Light orange
    elif 0 <= nsm <= 10:
        return '#FEB24C'  # Yellow-orange
    elif 10 < nsm <= 25:
        return '#FED976'  # Light yellow
    elif 25 < nsm <= 50:
        return '#C2E699'  # Light green
    elif 50 < nsm <= 100:
        return '#78C679'  # Medium green
    else:
        return '#238443'  # Dark green (strong accretion)


# === Ribbon Creator ===
def create_ribbons(gdf, nsm_col=None, color_func=None, fixed_color=None, width=30):
    polygons, colors, nsms = [], [], []
    for _, row in gdf.iterrows():
        line = row.geometry
        if line.is_empty or len(line.coords) < 2:
            continue
        try:
            offset = line.parallel_offset(width / 2, side='right', join_style=2)
            coords = list(line.coords) + list(offset.coords)[::-1]
            poly = Polygon(coords)
            polygons.append(poly)
            if color_func and nsm_col:
                colors.append(color_func(row[nsm_col]))
                nsms.append(row[nsm_col])
            else:
                colors.append(fixed_color)
                nsms.append(None)
        except:
            continue
    return polygons, colors, nsms

# === Load & Reproject Data ===
shoreline = gpd.read_file("../Final_shoreline1220.shp").to_crs(epsg=3857)
baseline = gpd.read_file("../Buffer/newbaseline12.shp").to_crs(epsg=3857)
transects = gpd.read_file("../DsasToArc/DSAS to Arc/Transect_rates1220.shp").to_crs(epsg=3857)
transects_70m = gpd.read_file("../QGIS_Export/70_rates.shp").to_crs(epsg=3857)

# === Clean NSM column ===
transects['NSM'] = pd.to_numeric(transects['NSM'], errors='coerce')
transects = transects.dropna(subset=['NSM'])
transects_70m['NSM'] = pd.to_numeric(transects_70m['NSM'], errors='coerce')
transects_70m = transects_70m.dropna(subset=['NSM'])

# === Create NSM Ribbon Polygons ===
polys, colors, nsms = create_ribbons(transects, nsm_col='NSM', color_func=get_fixed_nsm_color)
zones_gdf = gpd.GeoDataFrame({'NSM': nsms, 'color': colors, 'geometry': polys}, crs="EPSG:3857").to_crs(epsg=4326)



polys_70m, colors_70m, nsms_70m = create_ribbons(transects_70m, nsm_col='NSM', color_func=get_fixed_nsm_color)
zones_gdf_70m = gpd.GeoDataFrame({'NSM': nsms_70m, 'color': colors_70m, 'geometry': polys_70m}, crs="EPSG:3857").to_crs(epsg=4326)

shoreline = clean_for_folium(shoreline.to_crs(epsg=4326))
baseline = clean_for_folium(baseline.to_crs(epsg=4326))

# === Build Map ===
center = zones_gdf.geometry.union_all().centroid
m = folium.Map(
    location=[center.y, center.x],
    zoom_start=15,         # Start closer in
    max_zoom=20,           # Allow closer zoom levels
    min_zoom=8,            # Optional: restrict how far you can zoom out
    tiles=None
)




# == Load the plot ==
def generate_nsm_distribution_plot(nsm_values, save_path="nsm_distribution.png"):
    values = np.array(nsm_values)
    mean = np.mean(values)
    median = np.median(values)
    mode = float(pd.Series(values).mode().iloc[0]) if len(values) > 0 else 0
    std_dev = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot histogram
    ax.hist(values, bins=20, density=True, alpha=0.6, color='skyblue', label='NSM Histogram')

    # Plot normal curve
    x = np.linspace(min_val, max_val, 100)
    if std_dev > 0:
        normal_curve = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)
        ax.plot(x, normal_curve, color='darkblue', label='Normal Distribution')

    # Lines for mean, median
    ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    ax.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')

    # Table with summary stats
    stats_table = pd.DataFrame({
        'Statistic': ['Min', 'Max', 'Mean', 'Median', 'Std Dev'],
        'Value': [f"{min_val:.2f}", f"{max_val:.2f}", f"{mean:.2f}", f"{median:.2f}", f"{std_dev:.2f}"]
    })

    # Render table on plot
    cell_text = [[row.Statistic, row.Value] for row in stats_table.itertuples()]
    table = plt.table(cellText=cell_text, colLabels=["Metric", "Value"], cellLoc='center',
                      loc='upper right', bbox=[0.65, 0.65, 0.3, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    ax.set_title("Michigan shoreline NSM Distribution Across Transects")
    ax.set_xlabel("NSM (m)")
    ax.set_ylabel("Density")
    ax.legend(loc='upper left')
    plt.tight_layout()

    # Save or return image
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Plot saved to {save_path}")


def generate_nsm_base64(nsm_values):
    values = np.array(nsm_values)
    mean = np.mean(values)
    median = np.median(values)
    std_dev = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={'height_ratios': [3, 1]})

    # Plot histogram and normal curve
    ax1.hist(values, bins=20, color='skyblue', density=True, alpha=0.7, label="Histogram")
    x = np.linspace(min_val, max_val, 100)
    if std_dev > 0:
        ax1.plot(x, (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean)/std_dev)**2),
                 color='navy', label='Normal Curve')
    ax1.axvline(mean, color='red', linestyle='--', label=f"Mean: {mean:.2f}")
    ax1.axvline(median, color='green', linestyle='--', label=f"Median: {median:.2f}")

    ax1.set_title("Michigan NSM Distribution")
    ax1.set_xlabel("NSM (m)")
    ax1.set_ylabel("Density")
    ax1.legend()

    # Table with summary stats
    stats = pd.DataFrame({
        "Metric": ["Min", "Max", "Mean", "Median", "Std Dev"],
        "Value": [f"{min_val:.2f}", f"{max_val:.2f}", f"{mean:.2f}", f"{median:.2f}", f"{std_dev:.2f}"]
    })
    ax2.axis("off")
    table = ax2.table(cellText=stats.values, colLabels=stats.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_base64



img_base64 = generate_nsm_base64(zones_gdf["NSM"]) 




# === Base Map Layers ===
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Imagery', name='Esri Satellite Imagery'
).add_to(m)

folium.TileLayer(
    tiles='https://stamen-tiles.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}.png',
    attr='Stamen Toner Lite, OpenStreetMap',
    name='Toner Lite'
).add_to(m)

folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    name="OpenStreetMap",
    attr="OpenStreetMap contributors",
    max_zoom=20
).add_to(m)


folium.TileLayer('CartoDB Positron', name='CartoDB Positron').add_to(m)
Fullscreen().add_to(m)

# === Add Layer Data ===
folium.GeoJson(shoreline, name='Shoreline',
               style_function=lambda x: {'color': '#00FFFF', 'weight': 3}).add_to(m)

folium.GeoJson(baseline, name='Baseline',
               style_function=lambda x: {'color': '#FFD700', 'weight': 5, 'dashArray': '5, 5'}).add_to(m)

folium.GeoJson(zones_gdf, name="NSM Ribbon",
               style_function=lambda f: {
                   'fillColor': f['properties']['color'],
                   'color': f['properties']['color'],
                   'weight': 10,
                   'fillOpacity': 0.75
               },
               tooltip=GeoJsonTooltip(fields=['NSM'], aliases=['NSM (m):'])).add_to(m)

folium.GeoJson(zones_gdf_70m, name="70m NSM Transects", show = False,
               style_function=lambda f: {
                   'fillColor': f['properties']['color'],
                   'color': f['properties']['color'],
                   'weight': 6,
                   'fillOpacity': 0.6
               },
               tooltip=GeoJsonTooltip(fields=['NSM'], aliases=['NSM (m):'])).add_to(m)


# === Legend ===
legend_html = """
<style>
    #legend {
        position: fixed;
        bottom: 20px;
        left: 20px;
        width: 280px;
        background-color: white;
        z-index: 9999;
        padding: 10px;
        border: 2px solid grey;
        font-size: 13px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        opacity: 0;
        max-height: 0;
        overflow: hidden;
        transition: all 0.4s ease-in-out;
        border-radius: 8px;
    }

    #legend.visible {
        opacity: 1;
        max-height: 1000px; /* Arbitrary large height */
    }

    #toggleLegend {
        position: fixed;
        bottom: 250px;
        left: 20px;
        z-index: 9999;
        background-color: white;
        padding: 6px 12px;
        border: 2px solid grey;
        cursor: pointer;
        font-weight: bold;
        font-size: 13px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        border-radius: 6px;
    }
</style>

<div id="toggleLegend">Show Legend</div>

<div id="legend">
    <b>NSM Heatmap (m)</b><br>
    <span style="background:#800026; padding: 4px 10px; display:inline-block;"></span> ≤ -80 (Very strong erosion)<br>
    <span style="background:#BD0026; padding: 4px 10px; display:inline-block;"></span> -80 to -40<br>
    <span style="background:#E31A1C; padding: 4px 10px; display:inline-block;"></span> -40 to -20<br>
    <span style="background:#FC4E2A; padding: 4px 10px; display:inline-block;"></span> -20 to -5<br>
    <span style="background:#FD8D3C; padding: 4px 10px; display:inline-block;"></span> -5 to 0<br>
    <span style="background:#FEB24C; padding: 4px 10px; display:inline-block;"></span> 0 to 10<br>
    <span style="background:#FED976; padding: 4px 10px; display:inline-block;"></span> 10 to 25<br>
    <span style="background:#C2E699; padding: 4px 10px; display:inline-block;"></span> 25 to 50<br>
    <span style="background:#78C679; padding: 4px 10px; display:inline-block;"></span> 50 to 100<br>
    <span style="background:#238443; padding: 4px 10px; display:inline-block;"></span> > 100 (Strong accretion)
</div>

<script>
    const toggleButton = document.getElementById('toggleLegend');
    const legend = document.getElementById('legend');

    toggleButton.onclick = function () {
        legend.classList.toggle('visible');
        toggleButton.innerText = legend.classList.contains('visible') ? 'Hide Legend' : 'Show Legend';
    };
</script>
"""


distribution_legend = f"""
<style>
  #nsmDistToggle {{
    position: fixed;
    bottom: 20px;
    left: 360px;
    z-index: 9999;
    background-color: #ffffff;
    border: 1px solid #ccc;
    padding: 6px 14px;
    cursor: pointer;
    font-size: 14px;
    border-radius: 6px;
    box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.15);
    transition: background-color 0.3s ease;
  }}
  #nsmDistToggle:hover {{
    background-color: #f0f0f0;
  }}

  #nsmDistPanel {{
    position: fixed;
    bottom: 70px;
    left: 360px;
    width: 380px;
    max-height: 480px;
    overflow-y: auto;
    background-color: white;
    border: 1px solid #ccc;
    padding: 12px;
    z-index: 9998;
    display: none;
    transition: all 0.4s ease;
    border-radius: 8px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.25);
  }}

  #nsmDistPanel h4 {{
    font-size: 16px;
    margin: 0 0 10px;
    padding: 0;
  }}

  #nsmDistPanel img {{
    width: 100%;
    height: auto;
    border-radius: 4px;
    box-shadow: 0 0 4px rgba(0,0,0,0.1);
  }}
</style>

<div id="nsmDistToggle">📉 Show NSM Distribution</div>
<div id="nsmDistPanel">
  <h4>NSM Distribution</h4>
  <img src="data:image/png;base64,{img_base64}" alt="NSM Distribution Plot">
</div>

<script>
  document.getElementById('nsmDistToggle').onclick = function () {{
    var panel = document.getElementById('nsmDistPanel');
    if (panel.style.display === 'none' || panel.style.display === '') {{
      panel.style.display = 'block';
      this.textContent = '📉 Hide NSM Distribution';
    }} else {{
      panel.style.display = 'none';
      this.textContent = '📉 Show NSM Distribution';
    }}
  }};
</script>
"""




m.get_root().html.add_child(folium.Element(legend_html))
m.get_root().html.add_child(folium.Element(distribution_legend))


# === Finalize Map ===
folium.LayerControl().add_to(m)
generate_nsm_distribution_plot(zones_gdf["NSM"], save_path="Michigan_nsm_distribution.png")
m.save("index.html")
print("✅ Map saved as 'interactive_shoreline_nsm_ribbons.html'")




# === Save NSM summary to .txt file ===
nsm_stats = {
    "Dataset": "NSM Ribbon",
    "Total Transects": len(zones_gdf),
    "Eroding Transects (NSM < 0)": (zones_gdf["NSM"] < 0).sum(),
    "Accreting Transects (NSM > 0)": (zones_gdf["NSM"] > 0).sum()
}

nsm70_stats = {
    "Dataset": "70m NSM Ribbon",
    "Total Transects": len(zones_gdf_70m),
    "Eroding Transects (NSM < 0)": (zones_gdf_70m["NSM"] < 0).sum(),
    "Accreting Transects (NSM > 0)": (zones_gdf_70m["NSM"] > 0).sum()
}

summary_df = pd.DataFrame([nsm_stats, nsm70_stats])

summary_path = "nsm_summary_report.txt"
with open(summary_path, "w") as f:
    f.write("NSM Transect Summary Report\n")
    f.write("="*32 + "\n\n")
    f.write(summary_df.to_string(index=False) + "\n")

print(f"✅ NSM summary saved to: {summary_path}")



