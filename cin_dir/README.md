# Cinemática Directa

En este directorio se encuentra alojado el código de la cinemática directa.

Para ejecutar (situarse en el directorio donde se encuentra cin_dir.py):
```bash
python3 .\cin_dir.py parametros-manipulador5.json
```

En el fichero JSON se tiene que configurar de la siguiente manera:

- Poner tantos parámetros de Denavit-Hartenberg como articulaciones o puntos auxiliares se quiera

- orden_T, indica el orden de los índices para realizar las multiplicaciones de las matrices T, para saber que puntos están unidos a que puntos. Las bifurcaciones se indican con sublistas, y en caso de existir algún punto medio (sólo 1), se tiene que poner el parámetro EF indicando cuál es el índice del punto que se quiere como punto medio.

Se pondrán diversos ejemplos para que se entienda mejor.
