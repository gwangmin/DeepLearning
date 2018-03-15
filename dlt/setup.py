from setuptools import setup, find_packages

setup(name='dlt',
		version='1.0.0',
		url='https://github.com/gwangmin/DeepLearning/tree/master/dlt',
		author='gwangmin',
		author_email='ygm.gwangmin@gmail.com',
		license='MIT',
		description='Provide Deep Learning models.',
		long_discription=open('README.md','r').read(),
		packages=find_packages(),
		zip_safe=False,
		install_requires=[
		'tensorflow>=1.0.0','Keras>=2.0.2','numpy>=1.12.1','matplotlib>=2.0.0'
		]
		)
