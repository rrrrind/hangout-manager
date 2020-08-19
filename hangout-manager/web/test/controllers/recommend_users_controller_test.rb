require 'test_helper'

class RecommendUsersControllerTest < ActionDispatch::IntegrationTest
  setup do
    @recommend_user = recommend_users(:one)
  end

  test "should get index" do
    get recommend_users_url
    assert_response :success
  end

  test "should get new" do
    get new_recommend_user_url
    assert_response :success
  end

  test "should create recommend_user" do
    assert_difference('RecommendUser.count') do
      post recommend_users_url, params: { recommend_user: { user_id: @recommend_user.user_id } }
    end

    assert_redirected_to recommend_user_url(RecommendUser.last)
  end

  test "should show recommend_user" do
    get recommend_user_url(@recommend_user)
    assert_response :success
  end

  test "should get edit" do
    get edit_recommend_user_url(@recommend_user)
    assert_response :success
  end

  test "should update recommend_user" do
    patch recommend_user_url(@recommend_user), params: { recommend_user: { user_id: @recommend_user.user_id } }
    assert_redirected_to recommend_user_url(@recommend_user)
  end

  test "should destroy recommend_user" do
    assert_difference('RecommendUser.count', -1) do
      delete recommend_user_url(@recommend_user)
    end

    assert_redirected_to recommend_users_url
  end
end
