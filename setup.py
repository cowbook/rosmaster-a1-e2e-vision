from setuptools import find_packages, setup

package_name = "rosmaster_a1_e2e_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/e2e_driver.launch.py"]),
        (f"share/{package_name}/config", ["config/runtime.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cowbook",
    maintainer_email="cowbook@users.noreply.github.com",
    description="Minimal ROS 2 end-to-end visual driving MVP for Rosmaster A1.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "e2e_driver_node = rosmaster_a1_e2e_vision.node:main",
        ],
    },
)
