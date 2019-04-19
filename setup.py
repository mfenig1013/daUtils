from setuptools import setup

setup(name='dautils',
      version='0.1',
      description='a few data analysis tools',
      url='http://github.com/mfenig1013/dautils',
      author='Max Fenig',
      author_email='mfenig1013@gmail.com',
      license='MIT',
      packages=['dautils'],
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=['numpy', 'scipy', 'matplotlib', 'pandas'],
      zip_safe=False)