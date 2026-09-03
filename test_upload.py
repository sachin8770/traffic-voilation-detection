import cloudinary
import cloudinary.uploader

cloudinary.config(
  cloud_name = "dozg3fcta",
  api_key = "161944285762788",
  api_secret = "GzhRJ-3Hvl_1ETsp0_pVnIgKt2Q"
)

test_image = "./violations/violation_car_6_frame_4.jpg"
print(f"Testing Cloudinary Upload for {test_image}...")

try:
    res = cloudinary.uploader.upload(test_image)
    print("SUCCESS! Image uploaded to:")
    print(res.get("secure_url"))
except Exception as e:
    print("FAILED!")
    print(e)
