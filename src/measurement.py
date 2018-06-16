import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import *
from scipy import interpolate

VIZ = False

#1
height_varr_SMPL = (411, 3455) # only the biggest value
#2
shoulder_vertex_array_SMPL = (1881, 744, 3164, 4232, 5342)# open contour
#3
chest_vertex_array_SMPL = (6489, 943, 2954, 1253, 750, 4240, 4736, 6412) # closed contour
#4
waist_vertex_array_SMPL =  (1325, 4408, 4955, 4962, 4749, 6297, 3024, 1250, 1266, 1487, 1486, 3477)  # closed contour
#5
high_hips_vertex_array_SMPL = (6384, 6379, 6386, 6385, 5257, 6370, 6544, 1784, 3122, 2917, 2919, 2927, 2920, 2923) # closed contour
#6
hips_vertex_array_SMPL = (3116, 6540, 6509, 4920, 4984, 4348,
                          862, 1513, 1446, 3084, 3116) # closed contour
#7
spine_vertex_array_SMPL = (3164, 3012, 1755, 3028, 3017, 1784) # open contour
#8
leg_vertex_array_SMPL = (1454, 959, 1012, 3380) # open contour
#9
arm_vertex_array_SMPL = (1881, 3011, 1621, 2202) # open contour
#10
arm_girth_1_vertex_array_SMPL = (1830, 635, 634, 1509, 1844, 1403, 1545, 681,
                            1271, 717, 1892, 1889, 1883, 1874, 1830) # closed contour

#10
arm_girth_2_vertex_array_SMPL = (4917, 4916, 4912, 4830, 6454, 4207, 5331) # closed contour

#11
quads_vertex_array_SMPL = (3131, 3132, 1162, 1159, 1500, 1501, 1478,
                           1229, 1228, 1262, 833, 870, 1138, 3135, 3131) # closed contour
#12
front_top_body_varr_SMPL =  (4305, 4089, 6379) # # open contour
#13
fattest_belly_location_varr_SMPL = (4345, 4404, 4193, 4288, 4372, 6372, 2914,
                                    887, 800, 938, 679, 920, 1769, 4345) # closed contour

# except arm girth
POSE1 = [np.pi, 0, 0.00000000e+00, 0.00000000e+00
, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-2.90804397e-02
,-1.70349888e-01,-1.57339697e-01, 1.99229767e-01, 2.75563626e-02
,-3.95290994e-02, 5.01094797e-03,-3.17092170e-01, 2.69800883e-01
, 1.75386428e-01,-1.62688652e-01, 9.26191252e-02,-3.50764890e-01
,-2.18011090e-01, 7.95485223e-02, 3.92048179e-02, 2.70612905e-02
,-1.94993414e-02,-3.91080496e-01,-2.71549012e-02, 8.86272647e-02
, 3.80889405e-01, 1.24842131e-01,-3.70302367e-02,-3.54733494e-02
, 6.44505341e-02,-3.00053455e-01,-8.75686681e-01, 1.79999669e-01
, 2.59084868e-01, 8.74228274e-01, 0.00000000e+00, 0.00000000e+00
, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
,-8.46253867e-02,-4.24860208e-02, 2.07807234e-01, 7.93574001e-03
, 9.38797796e-02,-1.77043806e-01,-1.51839537e-01,-7.52624817e-02
,-2.31486025e-01,-1.25785288e-01, 1.34601422e-01, 2.43101565e-01]



def curve_length(X, Y, Z):
    'Calculate the length of 3d cureve'
    assert (len(X) == len(Y) and len(X) == len(Z)), "The array length must be the same"
    distance = 0
    for i in range(0, len(X) - 1):
        distance += np.sqrt((X[i + 1] - X[i])**2 + (Y[i + 1] - Y[i])**2 + (Z[i + 1] - Z[i])**2)
    return distance


def measure_height_2(sv):
    height = np.sqrt(np.sum((sv.r[411] - sv.r[3455])**2))
    return height


def measure_height(mesh, vertex_array):
    X = []
    Y = []
    Z = []
    for i in vertex_array:
        X.append(mesh[i][0])
        Y.append(mesh[i][1])
        Z.append(mesh[i][2])

    lenX = np.abs(X[1] - X[0])
    lenY = np.abs(Y[1] - Y[0])
    lenZ = np.abs(Z[1] - Z[0])
    height = max(lenX, lenY, lenZ)
    if(VIZ):
        print "The height is", height
    return height

def measure_part_open(mesh, vertex_array, name = 'NOT SPECIFIED', pk = 2,  VIZ = False):
    # Let us collect the coordinates of needed vertices into an array
    X = []
    Y = []
    Z = []
    for i in vertex_array:
        X.append(mesh[i][0])
        Y.append(mesh[i][1])
        Z.append(mesh[i][2])

    # the number of points for interpolation
    num_true_pts = 200
    # Interpolate
    tck, u = interpolate.splprep([X, Y, Z], s=2, k = pk)
    X_knots, Y_knots, Z_knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0, 1, num_true_pts)
    X_fine, Y_fine, Z_fine = interpolate.splev(u_fine, tck)

    if(VIZ):
        plt.close('all')
        f = plt.figure()
        ax = Axes3D(f)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        ax3d = f.gca(projection='3d')
        ax3d.set_aspect("equal")
        ax3d.plot(mesh.T[0], mesh.T[1],  mesh.T[2], 'k,')

        ax3d.plot(X, Y, Z, 'r')
        ax3d.plot(X_knots, Y_knots, Z_knots, 'go')
        ax3d.plot(X_fine, Y_fine, Z_fine)
        f.show()
        print "The measurement of ", name, " has been completed: "
        print "Interpolated distance = ", curve_length(X_fine, Y_fine, Z_fine)
        print "Raw distance = ", curve_length(X, Y, Z)
        #raw_input('look')

    return curve_length(X_fine, Y_fine, Z_fine)


def project_points(x, y, z, a, b, c):
    """
    Projects the points with coordinates x, y, z onto the plane
    defined by a*x + b*y + c*z = 1
    """
    vector_norm = a*a + b*b + c*c
    normal_vector = np.array([a, b, c]) / np.sqrt(vector_norm)
    point_in_plane = np.array([a, b, c]) / vector_norm

    points = np.column_stack((x, y, z))
    points_from_point_in_plane = np.subtract(points, point_in_plane)
    proj_onto_normal_vector = np.dot(points_from_point_in_plane,
                                     normal_vector)
    proj_onto_plane = (points_from_point_in_plane -
                       proj_onto_normal_vector[:, None]*normal_vector)

    return point_in_plane + proj_onto_plane


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def measure_part_closed(mesh, vertex_array, VIZ = False, name = 'NOT SPECIFIED'):
    # Let us collect the coordinates of needed vertices into an array
    Xdata = []
    Ydata = []
    Zdata = []
    for i in vertex_array:
        Xdata.append(mesh[i][0])
        Ydata.append(mesh[i][1])
        Zdata.append(mesh[i][2])

    Xdata.append(Xdata[0])
    Ydata.append(Ydata[0])
    Zdata.append(Zdata[0])

    # Decorator to count functions calls
    import functools
    def countcalls(fn):
        "decorator function count function calls "

        @functools.wraps(fn)
        def wrapped(*args):
            wrapped.ncalls +=1
            return fn(*args)

        wrapped.ncalls = 0
        return wrapped

    #  == METHOD 2 ==
    # Basic usage of optimize.leastsq
    from scipy import optimize
    # plotting functions
    from matplotlib import pyplot as p
    #p.close('all')


    # FIR A SURFACE
    def surface_point_distance(a, b, c, d):
        return  (np.multiply(Xdata, a) + np.multiply(Ydata, b) + np.multiply(Zdata, c) + d) / sqrt(a**2 + b**2 + c**2)

    a_m = 100.
    b_m = 100.
    c_m = 100.
    d_m = 100.


    def f_3(c):
        D = surface_point_distance(*c)
        return D


    distance_estimate = a_m, b_m, c_m, d_m
    params, ier = optimize.leastsq(f_3, distance_estimate,  maxfev = 50)

    a, b, c, d = params
    # create x,y
    xx, yy = np.mgrid[-0.5:0.5:20j, -0.5:0.5:20j]
    # calculate corresponding z
    zz = (-a*xx - b*yy - d) / c
    # calculate corresponding z
    zzz = (-a*xx - b*yy - 1) / c

    zzzz = 0

    vector_norm1 = a*a + b*b + c*c
    normal_vector1 = np.array([a, b, c]) / np.sqrt(vector_norm1)
    normal_vector2 = np.array([0, 0, 1])

    angle = np.pi/180. - angle_between(normal_vector1, normal_vector2)
    proj = project_points(Xdata, Ydata, Zdata, a, b, c)

    axis = np.cross(normal_vector1, normal_vector2)
    proj = dot(proj, rotation_matrix(axis, angle))
    xprj, yprj, zprj = proj.T


    proj2 = project_points(xprj, yprj, zprj, 0, 0, 1)
    xprj2, yprj2, zprj2 = proj2.T



    # FIR A SURFACE
    def ellipse_distance_to_point(xc, yc, a, b, alpha):
        # convert to from cartesian to polar
        xrt = (xprj2 - xc) * cos(alpha) - (yprj2 - yc) * sin(alpha) + xc
        yrt = (xprj2 - xc) * sin(alpha) + (yprj2 - yc) * cos(alpha) + yc

        r2 = sqrt((xrt - xc)**2 + (yrt - yc)**2)
        theta2 = np.arctan((yrt - yc)/ (xrt - xc))
        return a * b / sqrt((a*sin(theta2))**2 + (b*cos(theta2))**2) - r2

    x_m = mean(xprj2)
    y_m = mean(yprj2)
    a = 1.
    b = 1.
    alpha = 0.

    param = x_m, y_m, a, b, alpha

    def f_ellipse(c):
        R = ellipse_distance_to_point(*c)
        return R  #- R.mean()

    rez, ier = optimize.leastsq(f_ellipse, param,  xtol = 1.e-10, ftol = 1.e-16)
    x_c, y_c, a, b, alpha = rez
    R = ellipse_distance_to_point(x_c, y_c, a, b, alpha).mean()

    alpha = -alpha

    theta = np.linspace(0,2*np.pi,1000)
    xee = a * sin(theta) + x_c
    yee = b * cos(theta) + y_c

    xee2 = (xee - x_c) * cos(alpha) - (yee - y_c) * sin(alpha) + x_c
    yee2 = (xee - x_c) * sin(alpha) + (yee - y_c) * cos(alpha) + y_c

    length = curve_length(xee2, yee2, np.linspace(0, 0, len(xee2)))


    if(VIZ):
        plt.close('all')
        f = p.figure( facecolor='white')
        ax = Axes3D(f)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        ax3d = f.gca(projection='3d')
        ax3d.set_aspect("equal")
        ax3d.plot(Xdata, Ydata, Zdata, color='red')
        ax3d.plot(mesh.T[0], mesh.T[1],  mesh.T[2], 'k,')

        p.draw()
        f.show()


        f2 = p.figure(facecolor='white')
        ax = f2.gca()
        ax.plot(xprj, yprj)
        ax.plot([x_c], [y_c], 'ro')
        ax.plot(xee2,yee2)
        plt.xlim([-0.5,0.5])
        plt.ylim([-0.5,0.5])
        ax.grid()
        f2.show()

        print "The measurement of ", name, " has been completed: "
        print "Ellipse distance = ",  curve_length(xee2, yee2, np.linspace(0, 0, len(xee2)))
        print "Raw distance = ", curve_length(Xdata, Ydata, Zdata)    # The intersection
        #raw_input('look')

    return 0.5 * curve_length(Xdata, Ydata, Zdata) + 0.5 * curve_length(xee2, yee2, np.linspace(0, 0, len(xee2)))



def get_height(verts):
    max = np.max(np.abs(verts[411] - verts[3464]), axis=0)
    return max

import keypoints

def output_measurements(verts, VIZ=True, out=''):
    # measure
    mheightl = get_height(verts)
    mshoulderl = measure_part_open(verts, keypoints.ShoulderWidth, pk=2, VIZ=VIZ)
    mbreast = measure_part_closed(verts, keypoints.BreastCircumference, VIZ=VIZ)
    mwaist = measure_part_closed(verts, keypoints.WaistCircumference, VIZ=VIZ)
    mhips = measure_part_closed(verts, keypoints.HipCircumference, VIZ=VIZ)
    result = 100 * np.asarray([mshoulderl, mbreast, mwaist, mhips])
    return result



if __name__ == "__main__":
    verts = np.load('/home/king/Documents/measurement/release/22.npy')
    output_measurements(verts)
