require 'test_helper'

class UserDetailsControllerTest < ActionDispatch::IntegrationTest
  setup do
    @user_detail = user_details(:one)
  end

  test "should get index" do
    get user_details_url
    assert_response :success
  end

  test "should get new" do
    get new_user_detail_url
    assert_response :success
  end

  test "should create user_detail" do
    assert_difference('UserDetail.count') do
      post user_details_url, params: { user_detail: { d_eight: @user_detail.d_eight, d_five: @user_detail.d_five, d_four: @user_detail.d_four, d_one: @user_detail.d_one, d_seven: @user_detail.d_seven, d_six: @user_detail.d_six, d_three: @user_detail.d_three, d_two: @user_detail.d_two, user: @user_detail.user } }
    end

    assert_redirected_to user_detail_url(UserDetail.last)
  end

  test "should show user_detail" do
    get user_detail_url(@user_detail)
    assert_response :success
  end

  test "should get edit" do
    get edit_user_detail_url(@user_detail)
    assert_response :success
  end

  test "should update user_detail" do
    patch user_detail_url(@user_detail), params: { user_detail: { d_eight: @user_detail.d_eight, d_five: @user_detail.d_five, d_four: @user_detail.d_four, d_one: @user_detail.d_one, d_seven: @user_detail.d_seven, d_six: @user_detail.d_six, d_three: @user_detail.d_three, d_two: @user_detail.d_two, user: @user_detail.user } }
    assert_redirected_to user_detail_url(@user_detail)
  end

  test "should destroy user_detail" do
    assert_difference('UserDetail.count', -1) do
      delete user_detail_url(@user_detail)
    end

    assert_redirected_to user_details_url
  end
end
