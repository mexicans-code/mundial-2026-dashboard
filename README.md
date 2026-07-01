# Dashboard Apuestas Mundial 2026

Dashboard de predicciones generadas por modelo ML. Solo fines educativos.

## Opciones para compartir

### Opcion 1: GitHub Pages (recomendada, gratis)

1. Crea un repo en https://github.com/new (nombre: `mundial-2026-dashboard`)
2. En la terminal:
```bash
cd dashboard
git init
git add .
git commit -m "Dashboard inicial"
git remote add origin https://github.com/TU_USUARIO/mundial-2026-dashboard.git
git push -u origin main
```
3. Ve a Settings > Pages > Source: GitHub Actions (o deploy from main branch /docs)
4. Tu pagina en: `https://TU_USUARIO.github.io/mundial-2026-dashboard/`

### Opcion 2: Render.com (gratis, con Flask)

1. Sube a GitHub (mismos pasos)
2. Ve a https://render.com > New > Web Service
3. Conecta tu repo
4. Build Command: `pip install -r dashboard/requirements.txt`
5. Start Command: `cd dashboard && python app.py`
6. Listo

### Opcion 3: Local

```bash
cd dashboard
pip install -r requirements.txt
python app.py
```
Abre http://localhost:5000

## Actualizar datos

```bash
python update_dashboard.py
```
