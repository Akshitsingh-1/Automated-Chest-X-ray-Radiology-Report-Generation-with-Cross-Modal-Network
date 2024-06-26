$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();


    // Upload Preview
    function readURL(input) {
        var files = input.files;
        var input1 = files[0];
        var input2 = files[1];
        var preview1 = $("#imagePreview");
        var preview2 = $("#imagePreview1");
    
        var reader1 = new FileReader();
        reader1.onload = function (e) {
            // Display the selected image in the first preview element
            preview1.css('background-image', 'url(' + e.target.result + ')');
            preview1.hide();
            preview1.fadeIn(650);
        }
        reader1.readAsDataURL(input1);
    
        var reader2 = new FileReader();
        reader2.onload = function (e) {
            // Display the selected image in the second preview element
            preview2.css('background-image', 'url(' + e.target.result + ')');
            preview2.hide();
            preview2.fadeIn(650);
        }
        reader2.readAsDataURL(input2);
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                var responseParts = data.split(",");
                var res = responseParts[0];
                var word = responseParts[1];
                var keys = responseParts[2];
                var full_report = responseParts[3];
                if ($('#extractKeys').is(':checked') && $('#completeReport').is(':checked')){
                    $('#result').html('<strong><i>Resultant Report:</i> </strong><br>' + res + '<br><br>' + word + '<br>'
                     + keys+ '<br><br>' +'Translated Report:'+ '<br>' + full_report)
                }
                else if($('#extractKeys').is(':checked')){
                    $('#result').html('<strong><i>Resultant Report:</i> </strong><br>' + res + '<br><br>' + word + '<br>'
                     + keys+ '<br>')
                }
                else if($('#completeReport').is(':checked')){
                    $('#result').html('<strong><i>Resultant Report:</i> </strong><br>' + res + '<br><br>' +'Translated Report:'+ '<br>' + full_report)
                }
                else{
                    $('#result').html('<strong><i>Resultant Report:</i> </strong><br>' + res);
                }
                               
                console.log('Success!');
            },
        });
       
    });

});
