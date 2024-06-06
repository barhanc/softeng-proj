# Brief description

Simple browser application for exploratory data analysis (mainly PCA and unsupervised clustering)
written as a project for Software Engineering course at AGH University. We used only Python with
scikit-learn and nicegui libraries.

![](/docs/img1.png)
![](/docs/img2.png)
![](/docs/img3.png)

# Installation

```shell
$ git clone https://github.com/barhanc/softeng-proj.git
$ cd softeng-proj
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
(venv) $ python3 app.py
```

To package as a one-file executable browser application:
* make sure that arg `native` is set to `False` in `ui.run(...)` function in file `app.py`
* make sure that arg `reload` is set to `False` in `ui.run(...)` function in file `app.py`
* if you are building for macOS add the following lines to `app.py`
  ```python
  # macOS packaging support
  from multiprocessing import freeze_support  # noqa
  freeze_support()  # noqa
  ```
* execute
  ```shell
  $ nicegui-pack --onefile --name "myapp" app.py
  ```

For other possible packaging options see: 
[nicegui documentation](https://nicegui.io/documentation/section_configuration_deployment).