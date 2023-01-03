from setuptools import setup

package_name = 'ipm_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='ldavisiv@seas.upenn.edu',
    description='This package contains the publishers and CV functions for inverse perspective mapping (IPM)',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ipm_publisher_fast = ipm_pkg.ipm_publisher_fast:main',
            'ipm_publisher_dummy = ipm_pkg.ipm_publisher_dummy:main',
            'ipm_publisher_sim = ipm_pkg.ipm_publisher_sim:main',
        ],
    },
)
