package com.raven.dehazedemo;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.Intent;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.bumptech.glide.Glide;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import pub.devrel.easypermissions.EasyPermissions;

public class MainActivity extends AppCompatActivity implements UpLoadImgModule.CallBackListener {

    ImageView imgHaze;
    ImageView imgDehaze;
    Button btChoosePhoto;

    //相册请求码
    private static final int ALBUM_REQUEST_CODE = 1;

    // 权限集合
    private String[] permissions = {Manifest.permission.WRITE_EXTERNAL_STORAGE};

    // 图片上传模块
    private UpLoadImgModule upLoadImgModule;

    // ProgressDialog
    ProgressDialog pg;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // init view
        initViews();

        // init uploadModule
        upLoadImgModule = new UpLoadImgModule(this);

        // 建立图片放置目录
        File hazeDir = new File("/sdcard/DehazeDemo/HazeImgs");
        File dehazeDir = new File("/sdcard/DehazeDemo/DehazeImgs");
        if(!hazeDir.exists())
            hazeDir.mkdirs();
        if(!dehazeDir.exists())
            dehazeDir.mkdirs();
    }

    private void initViews() {
        imgHaze = findViewById(R.id.img_haze);
        imgDehaze = findViewById(R.id.img_dehaze);
        btChoosePhoto = findViewById(R.id.bt_choose_photo);

        pg = new ProgressDialog(this);

        btChoosePhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                callSystemAlbum();
            }
        });
    }

    /**
     * 调用相册，首次使用申请权限
     */
    private void callSystemAlbum() {
        if (EasyPermissions.hasPermissions(this, permissions)) {
            //已经打开权限
            usePhotoAlbum();
        } else {
            //没有打开相册权限、申请权限
            getPermission();
        }
    }

    private void usePhotoAlbum() {
        Intent intent = new Intent();
        intent.setAction(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, ALBUM_REQUEST_CODE);
    }


    /**
     * 获取权限
     */
    private void getPermission() {
        EasyPermissions.requestPermissions(this, "需要获取您的相册、照相使用权限", 1, permissions);
    }

    /**
     * 处理相机或者相册的回调
     *
     * @param requestCode 请求码
     * @param resultCode  结果码
     * @param data        传出data
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        // 相册回调处理
        if (requestCode == ALBUM_REQUEST_CODE && resultCode == RESULT_OK) {
            String photoPath = GetPhotoFromAlbum.getRealPathFromUri(this,
                    data != null ? data.getData() : null);

            Glide.with(this).load(photoPath).into(imgHaze);
            // 上传
            upLoadImgModule.uploadImg(photoPath);
            // 关闭button选项
            btChoosePhoto.setClickable(false);

            pg.setTitle("提示");
            pg.setMessage("去雾进行中...");
            pg.show();
        }
        super.onActivityResult(requestCode, resultCode, data);
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        //框架要求必须这么写
        EasyPermissions.onRequestPermissionsResult(requestCode, permissions, grantResults, this);
    }

    @Override
    public void onDehazeSuccess(final String dehazeImgPath) {
        Log.i("dehazeImgPath",dehazeImgPath);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                pg.cancel();
                btChoosePhoto.setClickable(true);
                Glide.with(MainActivity.this).load(dehazeImgPath).into(imgDehaze);
                // 通知图库更新
                notifyAlbum(dehazeImgPath);
            }
        });
    }

    @Override
    public void onDehazeError(final String msg) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                pg.cancel();
                btChoosePhoto.setClickable(true);
                Toast.makeText(MainActivity.this,msg,Toast.LENGTH_SHORT).show();
            }
        });
    }

    /**
     * 更新图库,photosPath 需要更新的图片的位置
     */
    private void notifyAlbum(String photosPath) {
        // 通知图库更新
        MediaScannerConnection.scanFile(MainActivity.this, new String[]{photosPath}, null,
                new MediaScannerConnection.OnScanCompletedListener() {
                    public void onScanCompleted(String path, Uri uri) {
                        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
                        mediaScanIntent.setData(uri);
                        sendBroadcast(mediaScanIntent);
                    }
        });

    }

}
