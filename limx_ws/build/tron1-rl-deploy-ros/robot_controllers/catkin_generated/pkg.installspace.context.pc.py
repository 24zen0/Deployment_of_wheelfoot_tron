# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include".split(';') if "${prefix}/include" != "" else []
PROJECT_CATKIN_DEPENDS = "roscpp;limxsdk_lowlevel;robot_common;robot_description;controller_interface;angles".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-lrobot_controllers".split(';') if "-lrobot_controllers" != "" else []
PROJECT_NAME = "robot_controllers"
PROJECT_SPACE_DIR = "/root/limx_ws/install"
PROJECT_VERSION = "0.0.1"
