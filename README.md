# Me Verifier – Verificador de Identidad Facial

Sistema que verifica si una imagen corresponde al rostro del usuario en este caso el mio
"yo vs un extraño".
Desarrollado en Python 3.11 con Flask, PyTorch y scikit-learn.



## Ejecución
```bash
python scripts/crop_faces.py
python scripts/embeddings.py
python train.py
python api/app.py
