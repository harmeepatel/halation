package main

import "core:fmt"
import "vendor:glfw"

main :: proc() {
	if !glfw.Init() {
		fmt.eprintln("Failed to initialize GLFW")
		return
	}

	defer glfw.Terminate()

	glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, 3)
	glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, 3)
	glfw.WindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
	glfw.WindowHint(glfw.RESIZABLE, glfw.TRUE)

	when ODIN_OS == .Windows {
		glfw.WindowHint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
	}

	window := glfw.CreateWindow(800, 600, "Odin GLFW Window", nil, nil)
	if window == nil {
		fmt.eprintln("Failed to create GLFW window")
		return
	}

	glfw.MakeContextCurrent(window)

	glfw.SwapInterval(1)

	fmt.println("Window created successfully! Close the window to exit.")

	for !glfw.WindowShouldClose(window) {
		glfw.PollEvents()


		glfw.SwapBuffers(window)
	}

	fmt.println("Window closed. Exiting.")
}
