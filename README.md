# Repositorio de contenedores para la prueba técnica

Este repositorio contiene las instrucciones de creación de contenedores usando docker-compose. En algunos casos hay Dockerfile asociados a estos archivos de configuración. Esto ocurre porque para algunos servicios es necesario modificar la imagen original para incluir funcionalidades extra.
Los archivos .env no están incluidos porque contienen información relacionada con la seguridad del servicio.
Es necesario crear .env en los servicios que lo necesiten (todos excepto *apache_spark* y *gpu_procesing*)

En estos archivos .env hay que incluir las variables de entorno necesarias para el correcto funcionamiento del servicio (revisar docker-compose.yml de cada uno).
> Nota: El archivo .env de Apache Airflow debe contener **AIRFLOW_UID=0**. La razón del ***0*** es porque este servicio en algunas ocasiones necesita ejecutar acciones en modo administrador dentro del propio contenedor.
> En caso de querer modificar los DAGs incluidos en la carpeta *apache_airflow/dags* es necesario modificar los permisos de escritura para el usuario: `$ sudo chown -R user:root apache_airflow/dags` donde *user* es el usuario que queramos.