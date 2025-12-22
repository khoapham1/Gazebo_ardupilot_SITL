from osgeo import gdal, osr
import numpy as np

# -------------------------------
# Cấu hình
# -------------------------------

# Kích thước grid (tăng càng nhiều, độ phân giải càng cao, file càng nặng)
image_size = (1000, 1000)   # 1000x1000 pixel ~ bản đồ chi tiết

# Tọa độ gốc (ví dụ HCMUTE, bạn thay bằng vị trí thực)
lon0, lat0 = 106.77282902, 10.85095073

# Độ phân giải ~ 0.00001 deg/pixel ~ ~1m tại vĩ độ này
# Bạn có thể chỉnh nhỏ hơn (0.000005) để zoom sâu hơn nữa
lon_step = 0.00001
lat_step = 0.00001

# -------------------------------
# Tạo dữ liệu lat/lon + elevation
# -------------------------------

lon = np.zeros(image_size, dtype=np.float64)
lat = np.zeros(image_size, dtype=np.float64)
elevation = np.zeros(image_size, dtype=np.float32)

for x in range(image_size[1]):  # cols
    for y in range(image_size[0]):  # rows
        lon[y, x] = lon0 + lon_step * x
        lat[y, x] = lat0 + lat_step * y
        # Thay bằng dữ liệu DEM thực tế nếu có
        elevation[y, x] = np.random.uniform(0, 50)  # ví dụ ngẫu nhiên 0–50m

# -------------------------------
# Tạo GeoTIFF
# -------------------------------

nx, ny = image_size
xmin, ymin, xmax, ymax = lon.min(), lat.min(), lon.max(), lat.max()
xres = (xmax - xmin) / float(nx)
yres = (ymax - ymin) / float(ny)

geotransform = (xmin, xres, 0, ymax, 0, -yres)

# Ghi file GeoTIFF
dst_ds = gdal.GetDriverByName('GTiff').Create(
    'elevation_highres.tif', ny, nx, 1, gdal.GDT_Float32
)

dst_ds.SetGeoTransform(geotransform)

# Hệ tọa độ WGS84
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
dst_ds.SetProjection(srs.ExportToWkt())

# Ghi band elevation
dst_ds.GetRasterBand(1).WriteArray(elevation)
dst_ds.FlushCache()
dst_ds = None

print("✅ GeoTIFF high-res đã được tạo: elevation_highres.tif")
