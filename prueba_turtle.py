import turtle

# Set key parameters
gravity = -0.005  # pixels/(time of iteration)^2
y_velocity = 1 / 10  # pixels/(time of iteration)
x_velocity = 0.25 / 10 # pixels/(time of iteration)
energy_loss = 0.95
width = 600
height = 800
# Set window and ball
window = turtle.Screen()
window.setup(width, height)
window.tracer(0)
window.bgcolor("black")
ball = turtle.Turtle()
ball.penup()
ball.color("white")
ball.shape("circle")

i=0
# Main loop
while i<50000:
    i+=1
    # Move ball
    ball.sety(ball.ycor() + y_velocity)
    ball.setx(ball.xcor() + x_velocity)
    # Acceleration due to gravity
    y_velocity += gravity
    # Bounce off the ground
    if ball.ycor() < -height:
        y_velocity = -y_velocity * energy_loss
        # Set ball to ground level to avoid it getting "stuck"
        ball.sety(-height)
    # Bounce off the walls (left and right)
    if ball.xcor() > width or ball.xcor() < -width:
        x_velocity = -x_velocity
    window.update()