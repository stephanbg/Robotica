# Cinemática Inversa

En este directorio se encuentra alojado el código de la cinemática inversa.

Para ejecutar (situarse en el directorio donde se encuentra ccdp3.py):
```bash
python3 .\ccdp3.py 10 10
```

x = 10 e y = 10 sería el punto objetivo, pero se puede situar en cualquier lugar del espacio 2D.

También, se puede modificar el fichero JSON, añadiendo articulaciones indicando:

- El tipo, si es de `revolucion` o `prismatica`
  
- Si es de `revolucion`:

  - Indicar: `th`, `min_th`, `max_th`, `a`

- Si es de `prismatica`:

  - Indicar: `a`, `min_a`, `max_a`
