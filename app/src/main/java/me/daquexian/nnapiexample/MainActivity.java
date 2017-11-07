package me.daquexian.nnapiexample;

import android.Manifest;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.InputStream;
import java.util.List;

import pub.devrel.easypermissions.EasyPermissions;

public class MainActivity extends AppCompatActivity
    implements EasyPermissions.PermissionCallbacks {

    @SuppressWarnings("unused")
    private static final String TAG = "NNAPI Example";
    private static final int PICK_IMAGE = 123;
    private static final int INPUT_LENGTH = 28;

    String[] perms = {Manifest.permission.READ_EXTERNAL_STORAGE};

    private TextView textView;
    private Button button;
    private ImageView imageView;
    private Bitmap selectedImage;

    static {
        OpenCVLoader.initDebug();
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textView = findViewById(R.id.text);
        button = findViewById(R.id.button);
        imageView = findViewById(R.id.imageView);

        textView.setText(R.string.welcome_message);
        button.setText(R.string.button_text);
        imageView.setScaleType(ImageView.ScaleType.CENTER_INSIDE);

        if (EasyPermissions.hasPermissions(this, perms)) {
            initListener();

            initModel(getAssets());
        } else {
            // Do not have permissions, request them now
            EasyPermissions.requestPermissions(this, "Please grant",
                    321, perms);
        }
    }

    private void initListener() {
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, PICK_IMAGE);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE && resultCode == RESULT_OK && data.getData() != null) {
            if (selectedImage != null && !selectedImage.isRecycled()) {
                selectedImage.recycle();
            }

            final InputStream imageStream;
            try {
                imageStream = getContentResolver().openInputStream(data.getData());
                selectedImage = BitmapFactory.decodeStream(imageStream);

                imageView.setImageBitmap(selectedImage);

                float[] inputData = getInputData(selectedImage);

                int predictNumber = predict(inputData);
                textView.setText(getResources().getString(R.string.predict_text, predictNumber));

            } catch (Exception e) {
                e.printStackTrace();
                textView.setText(e.getMessage());
            }
        }
    }

    private float[] getInputData(Bitmap bitmap) {
        Mat imageMat = new Mat();
        Mat inputMat = new Mat();

        Utils.bitmapToMat(bitmap, imageMat);

        // convert the image to 28 * 28, grayscale, 0~1, and smaller means whiter
        Imgproc.cvtColor(imageMat, imageMat, Imgproc.COLOR_RGBA2GRAY);
        imageMat = centerCropAndScale(imageMat, INPUT_LENGTH);
        imageMat.convertTo(imageMat, CvType.CV_32F, 1. / 255);
        Core.subtract(Mat.ones(imageMat.size(), CvType.CV_32F), imageMat, inputMat);

        float[] inputData = new float[inputMat.width() * inputMat.height()];

        inputMat.get(0, 0, inputData);

        return inputData;
    }

    private Mat centerCropAndScale(Mat mat, int length) {
        Mat _mat = mat.clone();
        if (_mat.height() > _mat.width()) {
            _mat = new Mat(_mat, new Rect(0, (_mat.height() - _mat.width()) / 2, _mat.width(), _mat.width()));
            Imgproc.resize(_mat, _mat, new Size(length, length));
        } else {
            _mat = new Mat(_mat, new Rect((_mat.width() - _mat.height()) / 2, 0, _mat.height(), _mat.height()));
            Imgproc.resize(_mat, _mat, new Size(length, length));
        }
        return _mat;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        clearModel();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        // Forward results to EasyPermissions
        EasyPermissions.onRequestPermissionsResult(requestCode, permissions, grantResults, this);
    }

    @Override
    public void onPermissionsGranted(int requestCode, List<String> perms) {
        recreate();
    }

    @Override
    public void onPermissionsDenied(int requestCode, List<String> perms) {
        finish();
    }

    public native void initModel(AssetManager assetManager);
    public native int predict(float[] data);
    public native int clearModel();
}
