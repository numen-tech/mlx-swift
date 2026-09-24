#!/bin/sh

BUILD_DIR=`xcodebuild -configuration Debug -showBuildSettings -scheme MLX | grep 'BUILT_PRODUCTS_DIR = /' | sed -e 's/^[^=]*= //g' | head -1`

# rpath points to PackageFrameworks so link it to the built products
(cd $BUILD_DIR/PackageFrameworks; ln -s ../*.framework .)

# run every bundle in MLX.xctestplan, and fail if any of them fails
status=0
xcrun xctest "$BUILD_DIR/MLXTests.xctest" || status=1
xcrun xctest "$BUILD_DIR/MLXIntegrationTests.xctest" || status=1
exit $status
