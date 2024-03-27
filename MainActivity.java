package com.example.fishclassify;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.fishclassify.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    // Deklarasi elemen UI
    Button selectBtn, captureBtn, predictBtn;
    TextView result, confidence;
    ImageView imageView;
    Bitmap bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Inisialisasi elemen UI
        selectBtn = findViewById(R.id.selectBtn);
        captureBtn = findViewById(R.id.captureBtn);
        predictBtn = findViewById(R.id.predictBtn);
        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);

        // Meminta izin kamera
        getPermission();

        // Memberikan aksi klik pada tombol pemilihan gambar dari galeri
        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openGallery();
            }
        });

        // Memberikan aksi klik pada tombol pemotretan gambar
        captureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                captureImage();
            }
        });

        // Memberikan aksi klik pada tombol prediksi
        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Memproses prediksi setelah gambar dipilih atau difoto
                processPrediction();
            }
        });
    }

    // Membuka galeri untuk memilih gambar
    private void openGallery() {
        Intent intent = new Intent();
        intent.setAction(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(intent, 10);
    }

    // Membuka kamera untuk pemotretan gambar
    private void captureImage() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, 12);
    }

    // Memproses prediksi menggunakan model TensorFlow Lite
    private void processPrediction() {
        // Menyesuaikan ukuran gambar
        bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

        try {
            // Menginisialisasi model TensorFlow Lite
            Model model = Model.newInstance(getApplicationContext());

            // Membuat input untuk model
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // Mendapatkan nilai piksel gambar
            int[] intValues = new int[224 * 224];
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
            int pixel = 0;

            // Iterasi setiap piksel dan ekstrak nilai R, G, dan B
            for (int i = 0; i < 224; i++) {
                for (int j = 0; j < 224; j++) {
                    int val = intValues[pixel++]; // RGB

                    // Ekstrak nilai asli komponen warna: Merah (R), Hijau (G), dan Biru (B)
                    byteBuffer.putFloat(((val >> 16) & 0xFF)); // Komponen Merah
                    byteBuffer.putFloat(((val >> 8) & 0xFF)); // Komponen Hijau
                    byteBuffer.putFloat((val & 0xFF)); // Komponen Biru
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Menjalankan inferensi model dan mendapatkan hasilnya
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();


            String[] classes = {"Cakalang", "Tongkol"}; // Kelas prediksi
            int maxPos = 2; // inisialisasi 3 kelas (kelas 0 = cakalang, kelas 1 = Tongkol, dan ditambahkan  Kelas 2 = "Unknown")
            float maxConfidence = 0.600f; // inisialisasi ambang batas probabilitas(60%)

            // Mencari kelas dengan probabilitas tertinggi dan menggunakan ambang batas probabilitas(60%)
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) { //Jika nilai probabilitas lebih dari ambang batas probabilitas(60%)
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            // Menampilkan hasil prediksi
            if (maxPos <= 1) { // Jika nilai maxPos kurang dari atau sama dengan 1 (kelas 0 atau kelas 1)
                result.setText(classes[maxPos]);
            } else { //Jika nilai probabilitas Kurang dari ambang batas probabilitas(60%) (kelas 2 = "Unknown")
                result.setText("Unknown");
            }

            // Menampilkan nilai probabilitas untuk setiap kelas prediksi
            String s = "";
            for (int i = 0; i < classes.length; i++) {
                s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100);
            }

            confidence.setText(s);

            // Melepaskan sumber daya model jika tidak lagi digunakan
            model.close();
        } catch (IOException e) {
            // Menangani pengecualian
            e.printStackTrace();
        }
    }

    // Meminta izin kamera
    private void getPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, 11);
            }
        }
    }

    // Menanggapi hasil permintaan izin
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == 11) {
            if (grantResults.length > 0) {
                if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                    this.getPermission();
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    // Menanggapi hasil aktivitas pemilihan atau pemotretan gambar
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 10) {
            // Mendapatkan gambar dari galeri
            if (data != null) {
                Uri uri = data.getData();
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    imageView.setImageBitmap(bitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        } else if (requestCode == 12) {
            // Mendapatkan gambar dari hasil pemotretan
            bitmap = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(bitmap);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}